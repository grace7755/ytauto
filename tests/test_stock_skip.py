"""Film-wide memory of irrelevant stock (fix A2).

Measured on video-4 (Russian firearms, 127 shots): one Mixkit waterfall was fetched, cut and AI-judged for 53 shots;
replaying the ledger, skipping an item after 2 low-relevance rejects on OTHER shots skips 206 of 827 stock/still
attempts and loses 0 accepted clips (scripts/measure_attempts.py)."""
from __future__ import annotations

from ytauto.config import Settings
from ytauto.models import Brief, Candidate, ClipScore, Decision, Shot
from ytauto.select import Selector
from ytauto.cache import Cache
from ytauto.state import Project


def _sel(tmp_path, shots, **kw):
    s = Settings(cache_dir=tmp_path / "cache", projects_dir=tmp_path / "p", library=False, **kw)
    proj = Project(tmp_path / "p" / "demo")
    return Selector(s, proj, Cache(tmp_path / "cache"), None, None, {}, list(shots)), proj


def _shot(i, subject="sniper rifle", entities=()):
    return Shot(id=i, segment=i, start=i * 4.0, end=i * 4.0 + 4.0, text=f"line {i}",
                brief=Brief(subject=subject, entities=list(entities)))


def _reject(proj, shot, source_id="2213", provider="mixkit", rel=5.0, status="rejected"):
    proj.log_decision(Decision(shot=shot, status=status, provider=provider, source_id=source_id, url="u",
                               score=ClipScore(relevance=rel, total=30)))


CLIP = Candidate(provider="mixkit", source_id="2213", url="u", title="waterfall in a forest")


def test_an_item_judged_irrelevant_on_two_other_shots_is_skipped_for_a_third(tmp_path):
    shots = [_shot(i) for i in range(4)]
    sel, proj = _sel(tmp_path, shots)
    sel._decide(shots[0], CLIP, "rejected", score=ClipScore(relevance=5.0, total=20))
    assert not sel._skip_irrelevant(CLIP, shots[2]), "one reject is not enough"
    sel._decide(shots[1], CLIP, "rejected", score=ClipScore(relevance=5.0, total=20))
    assert sel._skip_irrelevant(CLIP, shots[2])


def test_a_rejection_on_the_same_shot_does_not_count_against_that_shot(tmp_path):
    shots = [_shot(i) for i in range(3)]
    sel, proj = _sel(tmp_path, shots)
    _reject(proj, 0)
    _reject(proj, 1)
    # shot 1 is one of the two: only ONE other shot (0) counts for it, so it is not skipped by this rule
    assert not sel._skip_irrelevant(CLIP, shots[1])


def test_the_item_stays_available_when_its_title_names_this_shots_subject(tmp_path):
    shots = [_shot(0), _shot(1), _shot(2, subject="waterfall"), _shot(3, subject="Fushimi shrine gate")]
    sel, proj = _sel(tmp_path, shots)
    _reject(proj, 0)
    _reject(proj, 1)
    assert not sel._skip_irrelevant(CLIP, shots[2]), "a waterfall clip may be right for a waterfall shot"
    assert sel._skip_irrelevant(CLIP, shots[3])


def test_only_low_relevance_stock_counts_and_youtube_never_does(tmp_path):
    shots = [_shot(i) for i in range(4)]
    sel, proj = _sel(tmp_path, shots)
    _reject(proj, 0, rel=60.0)                           # a near miss is not "irrelevant"
    _reject(proj, 1, rel=80.0, status="accepted")
    _reject(proj, 2, provider="youtube", source_id="v1", rel=0.0)
    assert not sel._skip_irrelevant(CLIP, shots[3])
    sel._decide(shots[1], Candidate(provider="youtube", source_id="v1", url="u"), "rejected",
                score=ClipScore(relevance=0.0, total=0))
    assert "youtube:v1" not in sel._low_relevance(), "YouTube candidates are never remembered here"


def test_new_rejections_are_remembered_without_re_reading_the_ledger(tmp_path):
    shots = [_shot(i) for i in range(4)]
    sel, proj = _sel(tmp_path, shots)
    sel._low_relevance()                                   # built from an empty ledger
    sel._decide(shots[0], CLIP, "rejected", score=ClipScore(relevance=0.0, total=10))
    sel._decide(shots[1], CLIP, "rejected", score=ClipScore(relevance=10.0, total=20))
    assert sel._skip_irrelevant(CLIP, shots[2])


def test_the_rule_can_be_switched_off(tmp_path):
    shots = [_shot(i) for i in range(3)]
    sel, proj = _sel(tmp_path, shots, stock_skip_after=0)
    _reject(proj, 0)
    _reject(proj, 1)
    assert not sel._skip_irrelevant(CLIP, shots[2])


def test_a_restarted_run_rebuilds_the_memory_from_the_ledger(tmp_path):
    shots = [_shot(i) for i in range(3)]
    sel, proj = _sel(tmp_path, shots)
    _reject(proj, 0)
    _reject(proj, 1)
    again, _ = _sel(tmp_path, shots)                      # a new process (a redo)
    assert again._skip_irrelevant(CLIP, shots[2])
