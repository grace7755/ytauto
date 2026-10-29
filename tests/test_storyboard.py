"""Storyboard screening: the seek-bar preview sprites stand in for watching the whole video.

Measured 2026-10-29 against the URL screener on real candidates: ~12 s and ~3k tokens per
judgement instead of ~50 s and ~240k tokens, agreeing on 11/15. Three of the four disagreements
were storyboards rejecting videos the URL screener passed but which then failed after download
anyway. These tests cover the parts that are pure code: sprite slicing, padding, tile->time maps.
"""
import io

import pytest
from PIL import Image, ImageDraw

from ytauto.llm.router import JudgeError
from ytauto.models import Brief, Candidate, Shot
from ytauto.screen import screen, screen_storyboard
from ytauto.sources import storyboard as SB


def _sheet(rows, cols, tile=(160, 90), filled=None, seed=0):
    """A sprite sheet whose tiles are distinct, textured images; slots past `filled` are black."""
    import random

    rnd = random.Random(seed)
    img = Image.new("RGB", (cols * tile[0], rows * tile[1]), "black")
    d = ImageDraw.Draw(img)
    n = rows * cols if filled is None else filled
    for k in range(n):
        y, x = divmod(k, cols)
        x0, y0 = x * tile[0], y * tile[1]
        for _ in range(40):                      # texture, so it is not mistaken for padding
            cx, cy = x0 + rnd.randrange(tile[0]), y0 + rnd.randrange(tile[1])
            d.rectangle((cx, cy, cx + 12, cy + 8), fill=(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


class _Resp:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


@pytest.fixture()
def sprites(monkeypatch):
    """Two 3x3 sheets of 90 s each (10 s per tile); the second is only 4/9 filled, as on YouTube."""
    sheets = {"u0": _sheet(3, 3, seed=1), "u1": _sheet(3, 3, filled=4, seed=2)}
    monkeypatch.setattr(SB.requests, "get", lambda url, timeout=30: _Resp(sheets[url]))
    spec = {"rows": 3, "cols": 3, "width": 160, "height": 90, "fetched": __import__("time").time(),
            "fragments": [{"url": "u0", "duration": 90.0}, {"url": "u1", "duration": 40.0}]}
    return spec


def test_tiles_get_their_true_source_times(sprites):
    tiles = SB._tiles(sprites, duration=130.0)
    times = [t for t, _ in tiles]
    assert times[:3] == [0.0, 10.0, 20.0]
    assert times[9] == 90.0, "the second sheet starts where the first ended"
    assert max(times) < 130.0


def test_padding_in_the_last_sheet_is_dropped(sprites, tmp_path):
    """The prototype showed black padding tiles at the end of the contact sheet: the last sprite
    sheet is partly filled. Those slots are not video and must not reach the judge."""
    sb = SB.load("vid", sprites, 130.0, tmp_path)
    assert sb is not None
    assert len(sb.times) == 13, sb.times            # 9 + 4 real tiles, no padding
    assert sb.times[-1] == 120.0


def test_contact_sheet_is_cached_on_disk(sprites, tmp_path, monkeypatch):
    SB.load("vid", sprites, 130.0, tmp_path)
    monkeypatch.setattr(SB.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("refetched")))
    again = SB.load("vid", sprites, 130.0, tmp_path)
    assert again is not None and len(again.times) == 13


def test_long_videos_are_subsampled_to_one_sheet(monkeypatch, tmp_path):
    sheet = _sheet(5, 5, seed=3)
    monkeypatch.setattr(SB.requests, "get", lambda url, timeout=30: _Resp(sheet))
    spec = {"rows": 5, "cols": 5, "fetched": __import__("time").time(),
            "fragments": [{"url": f"u{i}", "duration": 250.0} for i in range(8)]}   # 200 tiles
    sb = SB.load("long", spec, 2000.0, tmp_path)
    assert len(sb.times) == SB.MAX_TILES
    assert sb.times[0] == 0.0 and sb.times[-1] > 1900


def test_expired_sprite_urls_are_refetched(sprites, tmp_path):
    stale = dict(sprites, fetched=0)                 # signed URLs from hours ago
    calls = []

    def refetch():
        calls.append(1)
        return sprites

    assert SB.load("vid", stale, 130.0, tmp_path, refetch_spec=refetch) is not None
    assert calls == [1]


def test_no_storyboard_returns_none(tmp_path):
    assert SB.load("none", None, 130.0, tmp_path) is None
    assert SB.spec_from_info({"formats": [{"format_id": "137"}]}) is None


def test_window_spans_the_tile_and_stays_in_the_video():
    sb = SB.Storyboard("v", b"", [0.0, 10.0, 20.0], 10.0, 25.0)
    assert sb.window(1) == (8.0, 16.0), "8 s: a 5 s shot plus room for scene detection, no more to download"
    assert sb.window(2) == (18.0, 25.0), "clipped to the end of the video"
    assert sb.window(0)[0] == 0.0
    assert sb.window(7) is None


def test_dense_tiles_of_a_short_video_still_give_a_usable_window():
    """Measured 2026-10-29: a 62 s video has a tile every ~0.97 s. Ending the window at the next
    tile made it 2.97 s, under screening's 3 s floor, so a 95% match was thrown away."""
    sb = SB.Storyboard("v", b"", [k * 0.97 for k in range(60)], 0.97, 62.0)
    a, b = sb.window(10)
    assert b - a >= 5.0 + 2.0, "room for a 5 s shot plus scene detection"
    assert sb.window(59)[1] == 62.0


# --- screening ---------------------------------------------------------------
SHOT = Shot(id=0, segment=0, start=0.0, end=5.0, text="Mount Fuji at sunrise.",
            brief=Brief(visual="Mount Fuji reflected in a lake.", entities=["Mount Fuji"],
                        youtube_queries=["mount fuji 4k"]))
CAND = Candidate(provider="youtube", source_id="abc", url="https://youtu.be/abc", duration=60)
SBOARD = SB.Storyboard("abc", b"jpeg", [float(t) for t in range(0, 60, 10)], 10.0, 60.0)


class _Judge:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.prompts = reply, exc, []

    def json_images(self, prompt, images, schema, **kw):
        self.prompts.append(prompt)
        if self.exc:
            raise self.exc
        return self.reply


def test_tiles_become_windows_in_the_judges_order():
    j = _Judge({"usable": True, "tiles": [3, 1, "2", 99, 3], "match": 90,
                "presenter_on_camera": False, "black_and_white": False})
    r = screen_storyboard(j, CAND, SHOT, SBOARD)
    assert [w["start"] for w in r["windows"]] == [28.0, 8.0, 18.0], "out-of-range and repeats dropped"
    assert [w["match"] for w in r["windows"]] == [90, 87, 84], "best first"
    assert r["usable"] is True


def test_ai_flag_is_never_taken_from_thumbnails():
    """The dedicated AI check on the HD clip decides that; a 160x90 thumbnail must not reject."""
    j = _Judge({"usable": True, "tiles": [1], "match": 80, "ai_or_animated": True,
                "presenter_on_camera": False, "black_and_white": False})
    assert screen_storyboard(j, CAND, SHOT, SBOARD)["ai_or_animated"] is False


def test_storyboard_chain_exhaustion_propagates():
    with pytest.raises(JudgeError):
        screen_storyboard(_Judge(exc=JudgeError("all judge providers failed")), CAND, SHOT, SBOARD)


def test_a_bad_reply_is_a_soft_failure():
    assert screen_storyboard(_Judge(exc=ValueError("no JSON")), CAND, SHOT, SBOARD) is None


def test_screen_falls_back_to_the_url_when_there_is_no_storyboard():
    class YT:
        def storyboard(self, c):
            return None

    class J(_Judge):
        def json_youtube(self, url, prompt, schema, **kw):
            self.url_called = True
            return {"usable": False, "windows": []}

    j = J()
    screen(j, CAND, SHOT, yt=YT(), mode="storyboard")
    assert getattr(j, "url_called", False)


def test_url_mode_never_touches_storyboards():
    class YT:
        def storyboard(self, c):
            raise AssertionError("storyboard used in url mode")

    class J(_Judge):
        def json_youtube(self, url, prompt, schema, **kw):
            return {"usable": False, "windows": []}

    screen(J(), CAND, SHOT, yt=YT(), mode="url")


def test_overlay_tiles_reorder_but_never_reject():
    """Thumbnails are too small to rule on a watermark: a flagged tile is only tried last."""
    j = _Judge({"usable": True, "tiles": [3, 1, 2], "overlay_tiles": [3], "match": 90,
                "presenter_on_camera": False, "black_and_white": False})
    r = screen_storyboard(j, CAND, SHOT, SBOARD)
    assert r["usable"] is True and r["overlay_suspect"] is False
    assert [w["start"] for w in r["windows"]] == [8.0, 18.0, 28.0], "clean tiles first, judge order kept"
    assert [w["overlay"] for w in r["windows"]] == [False, False, True]


def test_every_tile_flagged_makes_the_video_a_suspect_not_a_reject():
    j = _Judge({"usable": True, "tiles": [1, 2], "overlay_tiles": [1, "2"], "match": 80,
                "presenter_on_camera": False, "black_and_white": False})
    r = screen_storyboard(j, CAND, SHOT, SBOARD)
    assert r["usable"] is True and r["overlay_suspect"] is True and len(r["windows"]) == 2


def test_verdicts_cached_before_overlay_tiles_existed_still_work():
    j = _Judge({"usable": True, "tiles": [1], "match": 80,
                "presenter_on_camera": False, "black_and_white": False})
    r = screen_storyboard(j, CAND, SHOT, SBOARD)
    assert r["overlay_suspect"] is False and r["windows"][0]["overlay"] is False


def test_a_failed_section_download_is_retried_once(tmp_path, monkeypatch):
    from ytauto.config import Settings
    from ytauto.sources import youtube as Y

    attempts = []

    class DL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("ffmpeg exited with code 3436169992")
            out = tmp_path / "v_000000_000100.mp4"
            out.write_bytes(b"x" * 20_000)

    monkeypatch.setattr(Y.yt_dlp, "YoutubeDL", DL)
    monkeypatch.setattr(Y.time, "sleep", lambda s: None)
    yt = Y.YouTube.__new__(Y.YouTube)
    yt.s = Settings()
    c = Candidate(provider="youtube", source_id="v", url="https://youtu.be/v", duration=60)
    got = yt.download_section(c, 1.5, 8.5, tmp_path, pad=1.5)
    assert got is not None and len(attempts) == 2

    attempts.clear()
    (tmp_path / "v_000000_000100.mp4").unlink()
    monkeypatch.setattr(DL, "download", lambda self, urls: (attempts.append(1), 1 / 0)[1])
    assert yt.download_section(c, 1.5, 8.5, tmp_path, pad=1.5) is None and len(attempts) == 2, "gives up after one retry"
