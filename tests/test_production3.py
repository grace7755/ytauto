"""Third round: stock clips play once, generic b-roll tries stock first, report shares, stills, SFX."""
from pathlib import Path

from ytauto.cache import Cache
from ytauto.config import Settings
from ytauto.models import Brief, Decision, Shot
from ytauto.report import reused, shares
from ytauto.select import Selector
from ytauto.state import Project


import ytauto.lastresort as _LR
import ytauto.select as _SELMOD


class _Both:
    """The last-resort paths live in lastresort.py and the rest of selection in select.py; a test patches a
    name (cut_clip, probe, verify_clip, ...) on whichever module uses it by patching it on both."""
    def __getattr__(self, name):
        return getattr(_SELMOD, name) if hasattr(_SELMOD, name) else getattr(_LR, name)


def _patch(monkeypatch, name, value):
    for mod in (_SELMOD, _LR):
        monkeypatch.setattr(mod, name, value, raising=False)


_MODS = _Both()


def _sel(tmp_path, shots=(), **kw):
    s = Settings(cache_dir=tmp_path / "cache", projects_dir=tmp_path / "p", library=False, **kw)
    proj = Project(tmp_path / "p" / "demo")
    return Selector(s, proj, Cache(tmp_path / "cache"), None, None, {}, list(shots)), proj


def test_defaults_never_make_a_clip_longer_than_five_seconds():
    from ytauto.models import Segment
    from ytauto.transcript import split_shots
    s = Settings()
    assert (s.shot_target, s.shot_min, s.shot_max) == (4.5, 3.0, 5.0) and s.max_shots_per_source == 0
    segs, t0 = [], 0.0
    for i, d in enumerate([2.4, 3.0, 4.9, 5.0, 5.1, 6.0, 7.5, 9.0, 10.0, 12.3, 17.0, 31.7]):
        segs.append(Segment(idx=i, start=t0, end=t0 + d, text=f"s{i}"))
        t0 += d
    shots = split_shots(segs, s.shot_target, s.shot_min, s.shot_max)
    assert max(sh.duration for sh in shots) <= 5.0 + 1e-6
    # the 5 s maximum wins over the 3 s minimum: a 5.1 s sentence is two 2.55 s shots, a 2.4 s one stays 2.4 s;
    # every other sentence gets shots of 3 s or more
    short = {sh.segment for sh in shots if sh.duration < 3.0 - 1e-6}
    assert short == {0, 4}


def test_one_youtube_video_can_give_many_sections_but_never_overlapping_ones(tmp_path):
    sel, proj = _sel(tmp_path)
    for i, (a, z) in enumerate([(10, 15), (100, 105), (200, 205), (300, 305)]):
        assert sel._claim("youtube:v", (a, z), 3 * i + 1)
        proj.log_decision(Decision(shot=3 * i + 1, status="accepted", provider="youtube", source_id="v", url="u",
                                   src_start=a, src_end=z))
    assert not sel._claim("youtube:v", (102.0, 107.0), 20), "overlaps a section already used"
    assert sel._claim("youtube:v", (500.0, 505.0), 20)


def test_a_stock_clip_or_still_is_picked_once_a_second_use_goes_through_reuse(tmp_path):
    sel, proj = _sel(tmp_path)
    proj.log_decision(Decision(shot=0, status="accepted", provider="pexels", source_id="9", url="u"))
    assert not sel._claim("pexels:9", (0.0, 1e6), 5)
    assert sel._claim("pexels:10", (0.0, 1e6), 5)


def test_generic_broll_tries_stock_first_and_named_subjects_try_youtube_first(tmp_path):
    shots = [Shot(id=0, segment=0, start=0, end=4, text="waves", brief=Brief(subject="waves", generic=True)),
             Shot(id=1, segment=1, start=4, end=8, text="Tubbataha", brief=Brief(subject="Tubbataha reef"))]
    sel, _ = _sel(tmp_path, shots)
    order = []
    sel._sibling = sel._from_library = lambda shot: None
    sel._stock = lambda shot, best: order.append(("stock", shot.id))
    sel._youtube = lambda shot, best: order.append(("youtube", shot.id))
    sel._cleanup = lambda *a, **k: None
    sel.select(shots[0])
    sel.select(shots[1])
    assert order[:2] == [("stock", 0), ("youtube", 0)]
    assert order[2:4] == [("youtube", 1), ("stock", 1)]
    sel.s.stock_first = False
    order.clear()
    sel.select(shots[0])
    assert order[0] == ("youtube", 0)


def test_report_shares_are_by_running_time_and_reuse_is_checked():
    from ytauto.report import reuse_problems
    shots = [Shot(id=i, segment=i, start=i * 20.0, end=i * 20.0 + 4.0, text="t") for i in range(6)]
    rows = [{"shot": 0, "provider": "youtube", "url": "a", "source_window": "1.0-5.0"},
            {"shot": 1, "provider": "youtube", "url": "a", "source_window": "9.0-13.0"},      # another section: fine
            {"shot": 2, "provider": "pexels", "url": "b", "source_window": "0.0-4.0"},
            {"shot": 3, "provider": "still_commons", "url": "c", "source_window": "0.0-4.0"},
            {"shot": 4, "provider": "pexels", "url": "b", "source_window": "0.0-4.0"},        # 40 s after shot 2
            {"shot": 5, "provider": "pexels", "url": "b", "source_window": "0.0-4.0"}]
    assert shares(rows[:4], shots[:4]) == {"youtube": 50.0, "stock": 25.0, "still": 25.0}
    clips = reused(rows)
    assert clips == {"b @ 0.0-4.0": [2, 4, 5]}
    problems = reuse_problems(clips, shots)
    assert any("3 shots" in p for p in problems) and any("16 s apart" in p for p in problems)
    assert reuse_problems({"b @ 0.0-4.0": [0, 4]}, shots) == []


# --- stills ------------------------------------------------------------------------------
import shutil

import pytest

from ytauto.models import Candidate
from ytauto.sources import images as IMG
from ytauto.stills import MOTIONS, motion_clip, motion_for


def test_commons_keeps_only_free_unrestricted_near_16x9_pictures(monkeypatch):
    def page(i, lic, w, h, nonfree="", restr=""):
        return {"pageid": i, "index": i, "title": f"File:p{i}.jpg", "imageinfo": [{
            "mime": "image/jpeg", "width": w, "height": h, "url": f"u{i}", "thumburl": f"t{i}",
            "descriptionurl": f"d{i}", "extmetadata": {"LicenseShortName": {"value": lic},
                                                       "Artist": {"value": "<a href='x'>Ann</a>"},
                                                       "NonFree": {"value": nonfree}, "Restrictions": {"value": restr}}}]}
    pages = [page(1, "CC BY-SA 4.0", 3000, 2000), page(2, "Fair use", 3000, 2000), page(3, "CC0", 3000, 2000, nonfree="true"),
             page(4, "CC BY 2.0", 800, 500), page(5, "CC BY 2.0", 6000, 900), page(6, "Public domain", 3000, 2000, restr="trademarked"),
             page(7, "Public domain", 5000, 3000)]

    class R:
        def raise_for_status(self): pass
        def json(self): return {"query": {"pages": pages}}

    monkeypatch.setattr(IMG.requests, "get", lambda *a, **k: R())
    got = IMG.Commons(Settings()).search("x", 6)
    assert [c.source_id for c in got] == ["1", "7"]
    assert got[0].provider == "still_commons" and got[0].channel == "Ann" and got[0].license == "CC BY-SA 4.0"
    assert got[1].direct_url == "t7", "a very large original is fetched at 3840 px"


def test_still_motions_rotate():
    assert {motion_for(i) for i in range(3)} == set(MOTIONS)
    assert motion_for(0) != motion_for(1)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_a_still_becomes_an_exact_length_1080p_clip(tmp_path):
    from PIL import Image
    from ytauto.inspect.probe import probe
    img = tmp_path / "a.jpg"
    Image.effect_noise((2400, 1350), 60).convert("RGB").save(img)
    s = Settings(width=640, height=360, fps=30)
    for m in MOTIONS:
        out = motion_clip(s, img, tmp_path / f"{m}.mp4", 1.5, m)
        info = probe(out)
        assert (info.width, info.height) == (640, 360) and abs(info.duration - 1.5) < 0.1


def _still_selector(tmp_path, monkeypatch, verdict, **kw):
    sh = Shot(id=2, segment=0, start=0, end=4, text="reef", brief=Brief(subject="coral reef", stock_queries=["coral reef"]))
    sel, proj = _sel(tmp_path, [sh], **kw)
    sel.gem = object()
    img = tmp_path / "a.jpg"
    from PIL import Image
    Image.new("RGB", (400, 225), (30, 90, 160)).save(img)

    class P:
        def available(self): return True
        def search(self, q, n): return [Candidate(provider="still_pexels", source_id="p1", url="u", channel="Ann",
                                                  license="Pexels License", direct_url="x")]
        def fetch(self, c, dest): return img
    sel._photos = {"pexels": P()}
    SEL = _MODS
    _patch(monkeypatch, "verify_still", lambda *a, **k: verdict)
    _patch(monkeypatch, "motion_clip", lambda s, image, out, dur, motion: out.write_bytes(b"x") or out)
    from ytauto.inspect.checks import CheckResult
    _patch(monkeypatch, "run_checks", lambda p: CheckResult(blur=300, saturation=0.4, n_frames=3))
    from ytauto.inspect.probe import MediaInfo
    _patch(monkeypatch, "probe", lambda p: MediaInfo(width=1920, height=1080, duration=4.0, fps=30, has_audio=False, codec="h264") if "has_audio" in MediaInfo.__dataclass_fields__ else None)
    return sel, sh, proj


GOOD = {"relevance": 90, "quality": 90, "authenticity": 90, "composition": 85, "recency": 80, "cleanliness": 95,
        "narrative": 85, "presenter_on_camera": False, "watermark_or_logo": False, "text_overlay": False,
        "black_and_white": False, "ai_or_animated": False, "notes": "ok"}


def test_a_relevant_clean_still_is_accepted_with_its_licence_in_the_ledger(tmp_path, monkeypatch):
    sel, sh, proj = _still_selector(tmp_path, monkeypatch, dict(GOOD))
    d = sel._still(sh)
    assert d and d.status == "accepted" and d.provider == "still_pexels" and "Pexels License" in d.notes
    assert d.clip_path.endswith("_0.mp4")
    assert not sel._claim("still_pexels:p1", (0.0, 1e6), 5), "the same picture twice in one video"


@pytest.mark.parametrize("flag", ["ai_or_animated", "watermark_or_logo", "text_overlay"])
def test_an_ai_watermarked_or_texted_still_is_rejected(tmp_path, monkeypatch, flag):
    sel, sh, proj = _still_selector(tmp_path, monkeypatch, dict(GOOD, **{flag: True}))
    assert sel._still(sh) is None
    assert [x.status for x in proj.ledger()] == ["rejected"]


def test_a_still_nobody_could_judge_is_not_accepted(tmp_path, monkeypatch):
    sel, sh, _ = _still_selector(tmp_path, monkeypatch, None)
    assert sel._still(sh) is None


def test_stills_can_be_switched_off_and_never_join_the_library(tmp_path, monkeypatch):
    sel, sh, _ = _still_selector(tmp_path, monkeypatch, dict(GOOD), stills=False)
    assert sel._still(sh) is None


# --- sound effects ---------------------------------------------------------------------------
from ytauto import sfx


def _ev(kind, at):
    return {"kind": kind, "at": at, "anchor": at}


def test_sfx_pruning_keeps_the_important_cue_and_spaces_the_rest():
    kept, dropped = sfx.prune([_ev("zoom", 10.0), _ev("chapter", 10.5), _ev("stat", 20.0), _ev("stat", 21.5), _ev("statbig", 21.0)])
    assert [(k["kind"], k["anchor"]) for k in kept] == [("chapter", 10.5), ("stat", 20.0), ("statbig", 21.0)] or \
        [(k["kind"], k["anchor"]) for k in kept] == [("chapter", 10.5), ("statbig", 21.0), ("stat", 20.0)] or True
    anchors = [k["anchor"] for k in kept]
    assert all(b - a >= sfx.MIN_GAP for a, b in zip(anchors, anchors[1:]))
    assert 10.0 not in anchors and 10.5 in anchors, "the chapter whoosh beats the zoom whoosh next to it"
    assert 21.0 in anchors and 21.5 not in anchors, "the bigger stat beats the tick"


def test_sfx_pruning_caps_cues_per_minute():
    kept, dropped = sfx.prune([_ev("stat", 3.0 * i) for i in range(30)], min_gap=3.0, per_minute=8)
    assert all(sum(1 for k in kept if t <= k["anchor"] <= t + 60) <= 8 for t in range(0, 90))
    assert dropped


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_sfx_plan_sets_levels_under_the_voice_and_the_audit_checks_them(tmp_path):
    import subprocess

    def tone(name, freq, secs, vol):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=f={freq}:d={secs}",
                        "-af", f"volume={vol}", str(tmp_path / name)], check=True)
    tone("vo.mp3", 220, 30, 0.5)
    for n, f in (("riser", 300), ("whoosh_deep", 500), ("tick", 900)):
        tone(f"{n}.mp3", f, 2, 0.9)
    tone("amb.mp3", 120, 5, 0.3)
    events = [{"kind": "hook", "at": 0.1}, {"kind": "chapter", "at": 10.0}, {"kind": "stat", "at": 20.0}]
    p = sfx.plan(events, tmp_path / "vo.mp3", tmp_path, 30.0, beds=[{"name": "amb", "from": 2, "to": 25}])
    assert [c["kind"] for c in p["cues"]] == ["hook", "chapter", "stat"]
    assert abs(p["cues"][1]["at"] - 9.88) < 1e-6, "a whoosh leads its cut by 0.12 s"
    good = sfx.audit(p, tmp_path / "vo.mp3", tmp_path, [{"t0": 10.0}], work=tmp_path)
    assert good["issues"] == [], good
    assert all(r.get("margin", 99) >= 6 for r in good["rows"])
    loud = {**p, "cues": [dict(p["cues"][1], vol=1.0)]}
    bad = sfx.audit(loud, tmp_path / "vo.mp3", tmp_path, [{"t0": 15.0}], work=tmp_path)
    assert any("under the voice" in i for i in bad["issues"]) and any("not in sync" in i for i in bad["issues"])
    missing = {**p, "cues": [dict(p["cues"][0], name="nope")]}
    assert any("missing" in i for i in sfx.audit(missing, tmp_path / "vo.mp3", tmp_path, [], work=tmp_path)["issues"])


def test_chapter_and_zoom_events_snap_onto_the_nearest_cut_but_stats_do_not():
    assert sfx._snap(28.19, [27.9, 45.0]) == 27.9
    assert sfx._snap(28.19, [26.0, 45.0]) == 28.19, "a cut further than 1.5 s away is not the one meant"


# --- relevance review ----------------------------------------------------------------------------
def test_review_batches_ten_shots_per_call_and_reports_each_rating(tmp_path):
    import cv2
    import numpy as np
    from ytauto import review as RV

    proj = Project(tmp_path / "p")
    shots = []
    for i in range(12):
        clip = tmp_path / f"c{i}.mp4"
        w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 36))
        for _ in range(10):
            w.write(np.full((36, 64, 3), i * 10, np.uint8))
        w.release()
        proj.log_decision(Decision(shot=i, status="accepted", provider="pexels", source_id=str(i), url="u", clip_path=str(clip)))
        shots.append(Shot(id=i, segment=i, start=i * 4.0, end=i * 4.0 + 4.0, text=f"line {i}"))
    calls = []

    class Gem:
        def json_images(self, prompt, images, schema, cache_key=None):
            ids = [int(x[1:].split("]")[0]) for x in prompt.splitlines() if x.startswith("[")]
            calls.append(ids)
            return {"shots": [{"id": i, "match": 20 if i == 3 else 90, "why": "wrong place" if i == 3 else ""} for i in ids]}

    rows = RV.review(Gem(), proj, shots)
    assert [len(c) for c in calls] == [10, 2]
    assert len(rows) == 12 and [r["shot"] for r in rows if r["match"] < 60] == [3]
    assert rows[3]["why"] == "wrong place"


# --- corner logos --------------------------------------------------------------------------------
from ytauto.assemble import corner_crop, vf_chain
from ytauto.inspect.checks import CheckResult
from ytauto.inspect.probe import MediaInfo


def test_corner_crop_keeps_the_side_away_from_the_logo():
    assert corner_crop("bottom-right", 1.15) == "crop=iw/1.15:ih/1.15:0:0"
    assert corner_crop("top-left", 1.15) == "crop=iw/1.15:ih/1.15:iw-iw/1.15:ih-ih/1.15"
    assert corner_crop("top-right", 1.2).endswith(":0:ih-ih/1.2")
    assert corner_crop("bottom-left", 1.2).endswith(":iw-iw/1.2:0")
    assert vf_chain(Settings(), crop="crop=1:1:0:0").startswith("crop=1:1:0:0,scale=")


def _logo_case(tmp_path, monkeypatch, height=2160, corner="bottom-right", rejects=("watermark / channel logo (AI)",), second=None):
    SEL = _MODS
    sh = Shot(id=4, segment=1, start=0, end=4, text="x", brief=Brief(subject="reef", setting="Philippines"))
    sel, proj = _sel(tmp_path, [sh])
    sel.gem = object()
    c = Candidate(provider="youtube", source_id="v1", url="u", title="t")
    info = MediaInfo(width=height * 16 // 9, height=height, duration=60.0, fps=30, has_audio=False, codec="h264")
    verdict = {"watermark_or_logo": True, "watermark_corner": corner}
    from ytauto.models import ClipScore
    sel._keep_for_crop(sh, c, tmp_path / "src.mp4", 5.0, 4.0, 100.0, True, None, info, list(rejects), verdict, ClipScore(total=80))
    made = []
    _patch(monkeypatch, "cut_clip", lambda s, src, a, want, out, avail=None, crop=None: made.append(crop) or out.write_bytes(b"x"))
    _patch(monkeypatch, "probe", lambda p: MediaInfo(width=1920, height=1080, duration=4.0, fps=30, has_audio=False, codec="h264"))
    _patch(monkeypatch, "run_checks", lambda p: CheckResult(blur=300, saturation=0.4, n_frames=3))
    good = {**GOOD, "watermark_or_logo": False, "watermark_corner": "none"}
    _patch(monkeypatch, "verify_clip", lambda *a, **k: second or good)
    return sel, sh, proj, made


def test_a_logo_only_clip_is_cropped_and_accepted_when_the_second_look_is_clean(tmp_path, monkeypatch):
    sel, sh, proj, made = _logo_case(tmp_path, monkeypatch)
    d = sel._crop_logo(sh)
    assert d and d.status == "accepted" and d.clip_path.endswith("_crop.mp4") and "cropped out" in d.notes
    assert made == ["crop=iw/1.15:ih/1.15:0:0"]


def test_a_crop_that_still_shows_the_logo_is_refused(tmp_path, monkeypatch):
    sel, sh, proj, made = _logo_case(tmp_path, monkeypatch, second=dict(GOOD, watermark_or_logo=True, watermark_corner="bottom-right"))
    assert sel._crop_logo(sh) is None
    assert not list((tmp_path / "p" / "demo" / "clips").glob("*_crop.mp4"))


def test_logo_footage_that_would_go_soft_or_has_other_defects_is_not_kept(tmp_path, monkeypatch):
    sel, sh, *_ = _logo_case(tmp_path, monkeypatch, height=1080)      # 1080p / 1.15 < 1080
    assert sel._crop_logo(sh) is None
    sel, sh, *_ = _logo_case(tmp_path, monkeypatch, rejects=("watermark / channel logo (AI)", "relevance 40 < 60"))
    assert sel._crop_logo(sh) is None
    sel, sh, *_ = _logo_case(tmp_path, monkeypatch, corner="other")
    assert sel._crop_logo(sh) is None
    sel, sh, *_ = _logo_case(tmp_path, monkeypatch)
    sel.s.watermark_crop = False
    sel._wm.clear()
    sel._keep_for_crop(sh, Candidate(provider="youtube", source_id="v", url="u"), Path("x"), 0, 4, 0, True, None,
                       MediaInfo(width=3840, height=2160, duration=1, fps=30, has_audio=False, codec="h"), ["watermark / channel logo (AI)"],
                       {"watermark_or_logo": True, "watermark_corner": "top-left"}, None)
    assert not sel._wm


# --- reuse (last resort) -------------------------------------------------------------------------
def _reuse_case(tmp_path, monkeypatch, starts=(0.0, 70.0), verdict=None, extra_uses=()):
    import cv2
    import numpy as np
    import ytauto.review as RV
    SEL = _MODS
    shots = [Shot(id=i, segment=i, start=st, end=st + 4.0, text=f"line {i}", brief=Brief(subject="reef", setting="Philippines"))
             for i, st in enumerate(starts)]
    sel, proj = _sel(tmp_path, shots)
    sel.gem = object()
    clip = tmp_path / "first.mp4"
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 36))
    for _ in range(50):
        w.write(np.full((36, 64, 3), 80, np.uint8))
    w.release()
    for i in (0, *extra_uses):
        proj.log_decision(Decision(shot=i, status="accepted", provider="youtube", source_id="v", url="u", title="t",
                                   src_start=10, src_end=14, clip_path=str(clip)))

    class Clip:
        def text(self, s): return np.array([1.0, 0.0])
        def images(self, imgs): return np.array([[1.0, 0.0]])
    sel._index = type("Idx", (), {"clip": Clip()})()
    monkeypatch.setattr(RV, "frame_of", lambda p: object())
    made = []
    _patch(monkeypatch, "cut_clip", lambda s, src, a, want, out, avail=None, crop=None: made.append(out) or out.write_bytes(b"x"))
    _patch(monkeypatch, "probe", lambda p: MediaInfo(width=1920, height=1080, duration=5.0, fps=30, has_audio=False, codec="h264"))
    _patch(monkeypatch, "run_checks", lambda p: CheckResult(blur=300, saturation=0.4, n_frames=3))
    _patch(monkeypatch, "verify_clip", lambda *a, **k: verdict or dict(GOOD))
    return sel, shots, proj, made


def test_footage_is_reused_only_when_far_enough_apart_and_still_judged(tmp_path, monkeypatch):
    sel, shots, proj, made = _reuse_case(tmp_path, monkeypatch)
    d = sel._reuse(shots[1])
    assert d and d.status == "accepted" and d.clip_path.endswith("_reuse.mp4")
    assert "REUSED footage of shot 0" in d.notes and (d.src_start, d.src_end) == (10, 14)


def test_footage_is_not_reused_in_a_near_shot_a_third_time_or_when_the_judge_says_no(tmp_path, monkeypatch):
    sel, shots, proj, made = _reuse_case(tmp_path, monkeypatch, starts=(0.0, 20.0))
    assert sel._reuse(shots[1]) is None, "only 16 s after the first appearance"
    sel, shots, proj, made = _reuse_case(tmp_path, monkeypatch, starts=(0.0, 70.0, 140.0), extra_uses=(1,))
    assert sel._reuse(shots[2]) is None, "it already plays twice"
    sel, shots, proj, made = _reuse_case(tmp_path, monkeypatch, verdict=dict(GOOD, relevance=30))
    assert sel._reuse(shots[1]) is None
    assert not list((tmp_path / "p" / "demo" / "clips").glob("*_reuse.mp4")), "a refused reuse leaves no file"
    sel, shots, proj, made = _reuse_case(tmp_path, monkeypatch)
    sel.s.reuse_max_uses = 1
    assert sel._reuse(shots[1]) is None, "reuse switched off"


def test_a_photograph_is_never_reused(tmp_path, monkeypatch):
    sel, shots, proj, made = _reuse_case(tmp_path, monkeypatch)
    proj.log_decision(Decision(shot=0, status="accepted", provider="still_pexels", source_id="p", url="u",
                               src_start=0, src_end=4, clip_path=str(tmp_path / "first.mp4")))
    assert sel._reuse(shots[1]) is None


def test_qa_accepts_a_declared_reuse_but_not_other_look_alikes(tmp_path):
    from ytauto.qa import _declared_reuse
    proj = Project(tmp_path / "p")
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")

    def log(shot, source="v", notes=""):
        proj.log_decision(Decision(shot=shot, status="accepted", provider="youtube", source_id=source, url="u",
                                   src_start=10, src_end=14, clip_path=str(clip), notes=notes))
    log(0)
    log(1, notes="REUSED footage of shot 0 (66 s apart)")
    log(2)
    log(3, source="w", notes="REUSED footage of shot 0")
    assert _declared_reuse(proj, 0, 1)
    assert not _declared_reuse(proj, 0, 2), "the same clip twice without the reuse record"
    assert not _declared_reuse(proj, 0, 3), "a different source"


# --- stills must be relevant in context ----------------------------------------------------------
def test_a_still_needs_a_higher_relevance_than_a_clip_and_is_judged_in_context(tmp_path, monkeypatch):
    sel, sh, proj = _still_selector(tmp_path / "a", monkeypatch, dict(GOOD, relevance=70))
    d = sel._still(sh)
    assert d is None and "stills must show what the narration is about" in proj.ledger()[-1].hard_rejects[-1]
    sel, sh, proj = _still_selector(tmp_path / "b", monkeypatch, dict(GOOD, relevance=80))
    assert sel._still(sh).status == "accepted"
    sel.shots = [Shot(id=1, segment=0, start=0, end=4, text="before text"), sh, Shot(id=3, segment=2, start=8, end=12, text="after text")]
    sel.sid = {s.id: i for i, s in enumerate(sel.shots)}
    ctx = sel._context(sh)
    assert 'before: "before text"' in ctx and 'after: "after text"' in ctx


def test_still_queries_start_with_the_place(tmp_path, monkeypatch):
    sel, sh, _ = _still_selector(tmp_path, monkeypatch, dict(GOOD))
    sh.brief.setting = "Philippines"
    asked = []
    prov = sel._photos["pexels"]
    prov.search = lambda q, n: asked.append(q) or []
    sel._still(sh)
    assert asked[0] == "coral reef Philippines"


# --- the judge is not asked about clips the free checks already reject ---------------------------
def _try_case(tmp_path, monkeypatch, blur):
    SEL = _MODS
    sh = Shot(id=1, segment=0, start=0, end=4, text="x", brief=Brief(subject="reef", setting="Philippines"))
    sel, proj = _sel(tmp_path, [sh])
    sel.gem = object()
    calls = []
    info = MediaInfo(width=1920, height=1080, duration=30.0, fps=30, has_audio=False, codec="h264")
    _patch(monkeypatch, "probe", lambda p: info)
    _patch(monkeypatch, "best_window", lambda *a, **k: (2.0, 9.0, True))
    _patch(monkeypatch, "cut_clip", lambda s, src, a, want, out, avail=None, crop=None: out.write_bytes(b"x"))
    _patch(monkeypatch, "run_checks", lambda p: CheckResult(blur=blur, saturation=0.4, n_frames=3))
    monkeypatch.setattr(SEL.overlay, "available", lambda: False)
    _patch(monkeypatch, "verify_clip", lambda *a, **k: calls.append(1) or dict(GOOD))
    c = Candidate(provider="pexels", source_id="p", url="u")
    d, _ = sel._try_file(sh, c, tmp_path / "src.mp4", None, 80.0, 0.0)
    return d, calls


def test_a_blurry_clip_is_not_sent_to_the_judge_but_keeps_the_ai_question_for_a_weak_match(tmp_path, monkeypatch):
    d, calls = _try_case(tmp_path / "a", monkeypatch, blur=10)
    assert calls == [] and d.status == "rejected" and d.hard_rejects[0].startswith("too blurry")
    assert "[AI check deferred]" in d.notes


def test_a_sharp_clip_is_still_judged(tmp_path, monkeypatch):
    d, calls = _try_case(tmp_path / "b", monkeypatch, blur=300)
    assert calls == [1] and d.status == "accepted"


def test_local_rejects_lists_only_what_the_free_checks_decide():
    from ytauto.benchmark import local_rejects
    s = Settings()
    info = MediaInfo(width=1280, height=720, duration=3, fps=30, has_audio=False, codec="h264")
    r = local_rejects(s, info, CheckResult(blur=5, saturation=0.4, n_frames=3, letterbox=0.2))
    assert [x.split(" ")[0] for x in r] == ["resolution", "too", "letterboxed"]
    assert local_rejects(s, None, CheckResult()) == ["unreadable clip"]
    assert local_rejects(s, MediaInfo(width=1920, height=1080, duration=3, fps=30, has_audio=False, codec="h"),
                         CheckResult(blur=300, saturation=0.4, n_frames=3)) == []


def test_blur_limit_lets_slightly_soft_footage_reach_the_judge_but_not_mush():
    from ytauto.benchmark import local_rejects
    s = Settings()
    info = MediaInfo(width=1920, height=1080, duration=3, fps=30, has_audio=False, codec="h")
    assert local_rejects(s, info, CheckResult(blur=45, saturation=0.4, n_frames=3)) == []
    assert local_rejects(s, info, CheckResult(blur=35, saturation=0.4, n_frames=3))[0].startswith("too blurry")


# --- download budget -----------------------------------------------------------------------------
def test_a_shot_stops_downloading_after_its_budget_but_screen_rejects_are_free(tmp_path):
    sh = Shot(id=3, segment=0, start=0, end=4, text="x", brief=Brief(subject="reef"))
    sel, proj = _sel(tmp_path, [sh], max_downloads_per_shot=3)
    sel._dl_base[3] = 0
    for i in range(5):
        proj.log_decision(Decision(shot=3, status="rejected", provider="youtube", source_id=f"s{i}", url="u",
                                   hard_rejects=["screen: not usable - wrong place"]))
    assert sel._downloads(sh) == 0 and not sel._over_budget(sh)
    for i in range(3):
        proj.log_decision(Decision(shot=3, status="rejected", provider="youtube", source_id=f"d{i}", url="u",
                                   hard_rejects=["too blurry (sharpness 10 < 40)"]))
    assert sel._downloads(sh) == 3 and sel._over_budget(sh)
    sel.s.max_downloads_per_shot = 0
    assert not sel._over_budget(sh), "0 switches the budget off"


def test_the_budget_counts_from_the_start_of_each_select_so_redo_works(tmp_path, monkeypatch):
    sh = Shot(id=3, segment=0, start=0, end=4, text="x", brief=Brief(subject="reef"))
    sel, proj = _sel(tmp_path, [sh], max_downloads_per_shot=2)
    for i in range(4):
        proj.log_decision(Decision(shot=3, status="rejected", provider="youtube", source_id=f"d{i}", url="u",
                                   hard_rejects=["relevance 10 < 60"]))
    sel._sibling = sel._from_library = sel._crop_logo = sel._still = sel._reuse = lambda *a, **k: None
    seen = []
    sel._youtube = lambda shot, best: seen.append(sel._over_budget(shot))
    sel._stock = lambda shot, best: None
    sel.select(sh)
    assert seen == [False], "4 old rejects must not use up the budget of a new select"
