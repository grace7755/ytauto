"""Chain exhaustion must be distinguishable from bad footage.

During the 12-shot validation run of 2026-10-29 all four judge links were flapping (HTTP 500/403,
SSL handshake timeouts, WinError 10053, 503 "high demand", and a reasoning model truncated before
answering). `screen()` caught every exception alike and returned None, which the selector recorded
as "screen: screening call failed" - indistinguishable in the ledger from footage that was judged
and found wanting, and it pushed the shot towards the stock fallback for a reason that had nothing
to do with YouTube.
"""
import pytest

from ytauto.llm.router import JudgeError
from ytauto.models import Brief, Candidate, Shot
from ytauto.screen import screen

SHOT = Shot(id=0, segment=0, start=0.0, end=5.0, text="Mount Fuji at sunrise.",
            brief=Brief(visual="Mount Fuji reflected in a lake.", entities=["Mount Fuji"],
                        youtube_queries=["mount fuji 4k"]))
CAND = Candidate(provider="youtube", source_id="abc123", url="https://youtu.be/abc123", duration=600)


class _Judge:
    def __init__(self, exc):
        self.exc = exc

    def json_youtube(self, *a, **kw):
        raise self.exc


def test_chain_exhaustion_propagates():
    """A JudgeError means every provider is down - the caller has to know that."""
    with pytest.raises(JudgeError):
        screen(_Judge(JudgeError("all judge providers failed; last error: HTTP 500")), CAND, SHOT)


def test_other_failures_still_return_none():
    """A single malformed reply is this candidate's problem and stays a soft failure."""
    assert screen(_Judge(ValueError("no JSON in reply")), CAND, SHOT) is None


def test_unexpected_errors_do_not_escape():
    """Anything else (a socket error inside one provider, say) must not abort the whole run."""
    assert screen(_Judge(OSError("connection reset")), CAND, SHOT) is None


def test_unavailable_is_a_valid_decision_status_and_is_not_a_reject():
    """`unavailable` rows must not be counted as rejected candidates, or the report blames the
    footage for our outage."""
    from ytauto.models import Decision

    d = Decision(shot=0, status="unavailable", provider="youtube", source_id="x", url="u",
                 hard_rejects=["judge chain exhausted: HTTP 500"])
    assert d.status != "rejected"

    # ...and it must not look like usable footage to the resume logic either.
    assert d.status not in ("accepted", "weak")


def test_outage_becomes_a_blocker_without_aborting_the_run(tmp_path, monkeypatch):
    """One shot's outage must not kill the whole run: later shots may succeed once a link's
    circuit breaker reopens, and the finished ones must not be re-done."""
    from ytauto.config import load_settings
    from ytauto.pipeline import Pipeline

    s = load_settings(None)
    s.projects_dir = tmp_path
    p = Pipeline.__new__(Pipeline)          # skip __init__: no network, no judge
    p.s, p.shots, p.vo = s, [
        Shot(id=0, segment=0, start=0, end=5, text="a", brief=SHOT.brief),
        Shot(id=1, segment=1, start=5, end=10, text="b", brief=SHOT.brief),
    ], None
    from ytauto.state import Project

    p.proj = Project(tmp_path / "proj")
    p.cache, p.gem, p.yt, p.providers = None, None, None, {}

    calls = []

    class Sel:
        def __init__(self, *a, **kw):
            pass

        def select(self, shot):
            calls.append(shot.id)
            if shot.id == 0:
                raise JudgeError("all judge providers failed")
            return object()                 # shot 1 succeeds

    monkeypatch.setattr("ytauto.pipeline.Selector", Sel)
    blockers = Pipeline.select_all(p)

    assert sorted(calls) == [0, 1], "the run must continue past the outage"
    assert len(blockers) == 1
    assert "judge unavailable" in blockers[0]
    assert "not a footage problem" in blockers[0]
    assert "redo --shot 0" in blockers[0]
