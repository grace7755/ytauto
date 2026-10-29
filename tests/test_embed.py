"""Local CLIP pre-ranking: pure-code parts (no model download in tests)."""
import io

import numpy as np
from PIL import Image

from ytauto.cache import Cache
from ytauto.config import Settings
from ytauto.inspect.embed import query_text, sheet_tiles
from ytauto.models import Brief, Candidate, Shot
from ytauto.screen import screen
from ytauto.select import Selector
from ytauto.sources.storyboard import Storyboard
from ytauto.state import Project


def test_sheet_is_cut_back_into_its_tiles():
    sheet = Image.new("RGB", (8 * 240, 2 * 135), "white")
    buf = io.BytesIO()
    sheet.save(buf, "JPEG")
    tiles = sheet_tiles(buf.getvalue(), 13, 8)
    assert len(tiles) == 13
    assert tiles[0].width == 240 and tiles[0].height < 135, "the number label strip is cut off"


def test_query_is_the_required_subject_not_the_stacked_visual():
    """Measured: CLIP on the plain subject separated usable videos at AUC 0.74, on the full
    `visual` sentence 0.67."""
    b = Brief(subject="Mount Fuji", visual="A slow drone shot of Mount Fuji at dawn with mist.",
              entities=["Mount Fuji", "Lake Kawaguchi"])
    q = query_text(b)
    assert "Mount Fuji" in q and "drone" not in q and "Lake Kawaguchi" in q


class _Clip:
    def text(self, s):
        return np.array([1.0, 0.0])


class _Index:
    """Tile scores per video: 'b' matches best, 'c' is far below."""
    clip = _Clip()
    best = {"a": 0.27, "b": 0.30, "c": 0.20}

    def scores(self, sb, q):
        return np.array([self.best[sb.video_id], 0.1])


class _YT:
    def storyboard(self, c):
        return None if c.source_id == "n" else Storyboard(c.source_id, b"", [0.0, 10.0], 10.0, 20.0)


def _sel(tmp_path, **kw):
    sel = Selector(Settings(screen_top_k=3, **kw), Project(tmp_path / "p"), Cache(tmp_path / "c"),
                   _YT(), object(), {}, [])
    sel._index = _Index()
    return sel


SHOT = Shot(id=0, segment=0, start=0, end=5, text="t", brief=Brief(subject="s", visual="v"))


def _pool():
    return [Candidate(provider="youtube", source_id=v, url="u", text_score=t)
            for v, t in (("a", 90), ("n", 80), ("c", 70), ("b", 10))]


def test_rank_orders_by_clip_and_puts_no_storyboard_videos_last(tmp_path):
    got = _sel(tmp_path, prerank="rank", screen_mode="storyboard")._prerank(SHOT, _pool())
    assert [c.source_id for c in got] == ["b", "a", "c"], "top_k=3; the URL-only video did not make it"


def test_filter_drops_far_below_the_best_and_skips_url_only_in_storyboard_only(tmp_path):
    got = _sel(tmp_path, prerank="filter", prerank_margin=0.04, screen_mode="storyboard_only")._prerank(SHOT, _pool())
    assert [c.source_id for c in got] == ["b", "a"]


def test_storyboard_only_never_watches_the_url():
    class YT:
        def storyboard(self, c):
            return None

    class J:
        def json_youtube(self, *a, **kw):
            raise AssertionError("URL screening is off")

    r = screen(J(), Candidate(provider="youtube", source_id="x", url="u"), SHOT, yt=YT(), mode="storyboard_only")
    assert r["usable"] is False and "no storyboard" in r["reason"]
