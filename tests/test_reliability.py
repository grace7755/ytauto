"""Long-video reliability: planner batches, whole-video defects, and never accepting unjudged clips."""
import pytest

from ytauto.cache import Cache
from ytauto.config import Settings
from ytauto.inspect.verify import needs_ai_check, verify_clip
from ytauto.llm.router import JudgeError, QuotaExhausted
from ytauto.models import Brief, Candidate, Segment, Shot
from ytauto.planner import chunk_shots, plan
from ytauto.select import Selector
from ytauto.state import Project


def _shots(n_segments, per=2):
    out, i = [], 0
    for seg in range(n_segments):
        for v in range(per):
            out.append(Shot(id=i, segment=seg, start=i * 5.0, end=i * 5.0 + 5, text=f"seg {seg}", variant=v))
            i += 1
    return out


# --- planner batches -----------------------------------------------------------
def test_chunks_cover_every_shot_once_and_never_split_a_segment():
    shots = _shots(150, per=2)                          # a ~25 min video: 300 shots
    chunks = chunk_shots(shots, 40)
    assert [sh.id for c in chunks for sh in c] == list(range(300))
    for a, b in zip(chunks, chunks[1:]):
        assert a[-1].segment != b[0].segment
    assert max(len(c) for c in chunks) <= 41


def test_a_short_video_is_still_one_call():
    assert len(chunk_shots(_shots(6), 40)) == 1


class _PlanJudge:
    def __init__(self, fail_batch=None, exc=ValueError("bad reply")):
        self.calls, self.fail_batch, self.exc = 0, fail_batch, exc

    def json(self, prompt, schema, cache_key=None, **kw):
        self.calls += 1
        if self.calls == self.fail_batch:
            raise self.exc
        ids = [int(l[1:l.index("]")]) for l in prompt.splitlines() if l.startswith("[")]
        return {"topic": "Japan", "shots": [
            {"id": i, "visual": f"A wide view of Kyoto temple {i}.", "entities": ["Kyoto"],
             "youtube_queries": ["kyoto 4k"], "stock_queries": ["kyoto"]} for i in ids]}


def test_planner_batches_and_falls_back_only_for_a_failed_batch(tmp_path):
    shots = _shots(50, per=2)
    segs = [Segment(idx=k, start=k * 10.0, end=k * 10 + 10, text=f"Kyoto part {k}") for k in range(50)]
    j = _PlanJudge(fail_batch=2)
    briefs = plan(segs, shots, "Japan", Settings(plan_critic=False), Cache(tmp_path), j)
    assert j.calls == 3
    assert set(briefs) == {sh.id for sh in shots}
    assert "temple 0" in briefs[0].visual                         # batch 1: AI briefs
    assert "temple" not in briefs[50].visual                      # batch 2: heuristic fallback


def test_quota_during_planning_is_not_hidden_behind_heuristic_briefs(tmp_path):
    shots = _shots(2)
    segs = [Segment(idx=k, start=k * 10.0, end=k * 10 + 10, text="Kyoto") for k in range(2)]
    with pytest.raises(QuotaExhausted):
        plan(segs, shots, "Japan", Settings(), Cache(tmp_path),
             _PlanJudge(fail_batch=1, exc=QuotaExhausted("spent", until=1.0)))


# --- whole-video defects --------------------------------------------------------
def _selector(tmp_path):
    return Selector(Settings(), Project(tmp_path / "p"), Cache(tmp_path / "c"), None, None, {}, [])


def test_a_watermarked_video_is_skipped_for_every_later_shot(tmp_path):
    sel = _selector(tmp_path)
    c = Candidate(provider="youtube", source_id="vid", url="u")
    sel._note_defects(c, ["watermark / channel logo (AI)"])
    assert sel.bad_video(c)
    other = Selector(Settings(), Project(tmp_path / "p2"), Cache(tmp_path / "c"), None, None, {}, [])
    assert other.bad_video(c), "shared across projects through the cache"


def test_burnt_in_text_needs_two_strikes_and_relevance_none(tmp_path):
    sel = _selector(tmp_path)
    c = Candidate(provider="youtube", source_id="vid", url="u")
    sel._note_defects(c, ["relevance 40 < 60", "too blurry (sharpness 39 < 60)"])
    assert not sel.bad_video(c), "window problems say nothing about the rest of the video"
    sel._note_defects(c, ["burnt-in text / branding overlay (AI)"])
    assert not sel.bad_video(c), "one captioned scene is not a captioned video"
    sel._note_defects(c, ["burnt-in text / branding overlay (AI)"])
    assert sel.bad_video(c)


# --- verify ----------------------------------------------------------------------
def test_the_ai_check_is_skipped_only_when_it_cannot_change_the_outcome():
    assert not needs_ai_check({"watermark_or_logo": True, "relevance": 90}, 60)
    assert not needs_ai_check({"relevance": 40}, 60)
    assert needs_ai_check({"relevance": 90}, 60)
    assert needs_ai_check({"relevance": 90, "ai_or_animated": True}, 60), \
        "the combined verdict's AI flag is the one that needs a second opinion"


class _VerifyJudge:
    def __init__(self, exc=None, verdict=None):
        self.exc, self.verdict, self.ai_checks = exc, verdict, 0

    def json_video_file(self, *a, **kw):
        if self.exc:
            raise self.exc
        return dict(self.verdict)

    def confirm_ai_generated(self, *a, **kw):
        self.ai_checks += 1
        return False


SHOT = Shot(id=0, segment=0, start=0, end=5, text="t", brief=Brief(visual="v", entities=["e"]))


def test_verify_raises_on_an_outage_instead_of_passing_the_clip(tmp_path):
    with pytest.raises(JudgeError):
        verify_clip(_VerifyJudge(exc=JudgeError("all judge providers failed")), tmp_path / "x.mp4", SHOT, "id")


def test_verify_skips_the_ai_check_on_a_watermarked_clip(tmp_path):
    j = _VerifyJudge(verdict={"relevance": 90, "watermark_or_logo": True})
    verify_clip(j, tmp_path / "x.mp4", SHOT, "id", min_relevance=60)
    assert j.ai_checks == 0
    j = _VerifyJudge(verdict={"relevance": 90})
    verify_clip(j, tmp_path / "x.mp4", SHOT, "id", min_relevance=60)
    assert j.ai_checks == 1


# --- brief review ------------------------------------------------------------------
class _CriticJudge:
    """Plans a stacked, invented brief; the review call strips it."""

    def __init__(self, review=None, fail_review=False):
        self.review, self.fail_review, self.prompts = review, fail_review, []

    def json(self, prompt, schema, cache_key=None, **kw):
        self.prompts.append(prompt)
        if "checking B-roll briefs" in prompt:
            if self.fail_review:
                raise ValueError("no JSON")
            return self.review
        return {"topic": "Japan", "shots": [{
            "id": 0, "subject": "Hakuba ski slopes",
            "visual": "A snow groomer climbing Hakuba ski slopes at dawn in slow motion with mist.",
            "entities": ["Hakuba", "snow groomer"], "youtube_queries": ["hakuba ski 4k"],
            "stock_queries": ["ski slope"]}]}


SKI = [Segment(idx=0, start=0.0, end=5.0, text="Hakuba has some of the best powder snow in the world.")]


def test_review_strips_invented_details_and_keeps_the_queries(tmp_path):
    shots = [Shot(id=0, segment=0, start=0, end=5, text=SKI[0].text)]
    j = _CriticJudge(review={"shots": [{"id": 0, "subject": "Hakuba ski slopes",
                                         "visual": "A wide view of Hakuba ski slopes in fresh powder.",
                                         "entities": ["Hakuba"]}]})
    b = plan(SKI, shots, "Japan", Settings(), Cache(tmp_path), j)[0]
    assert "groomer" not in b.visual and b.entities == ["Hakuba"]
    assert b.youtube_queries[0] == "hakuba ski 4k", "search queries stay the planner's"
    assert "snow groomer" in j.prompts[1], "the reviewer sees the draft brief"


def test_a_failed_review_keeps_the_planners_brief(tmp_path):
    shots = [Shot(id=0, segment=0, start=0, end=5, text=SKI[0].text)]
    b = plan(SKI, shots, "Japan", Settings(), Cache(tmp_path), _CriticJudge(fail_review=True))[0]
    assert b.subject == "Hakuba ski slopes" and "groomer" in b.visual


def test_every_brief_has_a_subject_even_offline():
    shots = [Shot(id=0, segment=0, start=0, end=5, text=SKI[0].text)]
    b = plan(SKI, shots, "Japan", Settings(), None, None)[0]
    assert b.subject == "Hakuba"
