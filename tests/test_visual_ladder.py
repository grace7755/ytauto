"""Round 6 visual ladder, slice 1: visual kind per shot, the attempt budget, the category fallback, and disclosure.

`visual_ladder: false` must give the exact old shot order; these tests pin both orders."""
from __future__ import annotations

from pathlib import Path

import pytest

from ytauto.cache import Cache
from ytauto.config import Settings
from ytauto.llm.router import QuotaExhausted
from ytauto.models import Brief, Candidate, ClipScore, Decision, Shot
from ytauto.planner import classify_visual, generalise_brief
from ytauto.report import substitutions
from ytauto.select import Selector
from ytauto.state import Project


def _sel(tmp_path, shots=(), gem=None, providers=None, **kw):
    s = Settings(cache_dir=tmp_path / "cache", projects_dir=tmp_path / "p", library=False, **kw)
    proj = Project(tmp_path / "p" / "demo")
    return Selector(s, proj, Cache(tmp_path / "cache"), None, gem, providers or {}, list(shots)), proj


def _shot(i=0, text="line", **brief):
    brief.setdefault("subject", "PPK-20 submachine gun")
    return Shot(id=i, segment=i, start=i * 4.0, end=i * 4.0 + 4.0, text=text, brief=Brief(**brief))


def _record(sel, order, *names):
    for n in names:
        setattr(sel, n, (lambda name: lambda shot, *a, **k: order.append(name))(n))
    sel._sibling = sel._from_library = lambda shot: None
    sel._cleanup = lambda *a, **k: None


ALL_STEPS = ("_youtube", "_stock", "_still", "_crop_logo", "_category", "_reuse")


# --- classify_visual: rules first --------------------------------------------------------------------
def test_viewer_lines_become_quote_cards_and_measurements_spec_cards():
    shots = [_shot(0, "Tell us in the comments which rifle you would pick."),
             _shot(1, "The barrel is 32 inches long and weighs about 20 pounds."),
             _shot(2, "It costs $1,400 in the United States."),
             _shot(3, "The factory sits on the edge of Izhevsk.")]
    briefs = {sh.id: sh.brief for sh in shots}
    changed = classify_visual(shots, briefs)
    got = [(briefs[i].visual_kind, briefs[i].graphic) for i in range(4)]
    assert got == [("graphic", "quote"), ("graphic", "spec"), ("graphic", "spec"), ("footage", "")]
    assert len(changed) == 3


def test_the_models_choice_stands_when_no_rule_applies_and_bad_values_fall_back_to_footage():
    shots = [_shot(0, "Kalashnikov's newest pistol", visual_kind="photo"),
             _shot(1, "A quiet road", visual_kind="cartoon", graphic="spec"),
             _shot(2, "Three things matter", visual_kind="graphic", graphic=""),
             _shot(3, "Three things matter", visual_kind="graphic", graphic="list")]
    briefs = {sh.id: sh.brief for sh in shots}
    classify_visual(shots, briefs)
    assert (briefs[0].visual_kind, briefs[0].graphic) == ("photo", "")
    assert (briefs[1].visual_kind, briefs[1].graphic) == ("footage", ""), "a graphic kind needs visual_kind graphic"
    assert (briefs[2].visual_kind, briefs[2].graphic) == ("graphic", "quote")
    assert (briefs[3].visual_kind, briefs[3].graphic) == ("graphic", "list")


def test_a_year_or_plain_number_is_not_a_spec():
    shots = [_shot(0, "In 1991 the Soviet Union collapsed."), _shot(1, "Two men built it in a garage.")]
    briefs = {sh.id: sh.brief for sh in shots}
    classify_visual(shots, briefs)
    assert [briefs[i].visual_kind for i in range(2)] == ["footage", "footage"]


def test_the_new_brief_fields_do_not_change_the_cache_key():
    b = Brief(subject="x", visual="y")
    assert b.key() == b.model_copy(update={"visual_kind": "photo", "graphic": "spec", "category_of": "z"}).key()


# --- generalise_brief: one cached call -----------------------------------------------------------------
class _Gem:
    def __init__(self, answer=None, exc=None):
        self.answer, self.exc, self.calls = answer, exc, []

    def json(self, prompt, schema, **kw):
        self.calls.append((prompt, kw))
        if self.exc:
            raise self.exc
        return self.answer


def test_generalise_rewrites_to_the_category_and_marks_it():
    gem = _Gem({"subject": "submachine gun bolt cycling", "visual": "slow motion close up",
                "youtube_queries": ["submachine gun bolt slow motion", "smg review"], "stock_queries": ["gun close up"]})
    sh = _shot(0, "The PPK-20 uses a simple blowback action.", entities=["PPK-20"], generic=True, visual_kind="photo")
    nb = generalise_brief(gem, sh)
    assert nb.subject == "submachine gun bolt cycling" and nb.category_of == "PPK-20 submachine gun"
    assert nb.entities == [] and nb.visual_kind == "footage" and not nb.generic
    assert not any("review" in q.lower().split() for q in nb.youtube_queries), "presenter words are stripped"
    assert gem.calls[0][1]["cache_key"][0] == "generalise-v1" and gem.calls[0][1]["task"] == "plan"


def test_generalise_gives_up_quietly_but_never_hides_a_spent_quota():
    sh = _shot(0)
    assert generalise_brief(_Gem(exc=RuntimeError("boom")), sh) is None
    assert generalise_brief(_Gem({"subject": "PPK-20 submachine gun", "youtube_queries": [], "stock_queries": []}), sh) is None
    assert generalise_brief(None, sh) is None
    with pytest.raises(QuotaExhausted):
        generalise_brief(_Gem(exc=QuotaExhausted("all links spent", 0.0)), sh)


# --- the order of steps in select() ------------------------------------------------------------------
def test_ladder_off_keeps_the_exact_old_order(tmp_path):
    sh = _shot(0, visual_kind="photo")
    sel, _ = _sel(tmp_path, [sh], visual_ladder=False)
    order = []
    _record(sel, order, *ALL_STEPS)
    sel.select(sh)
    assert order == ["_youtube", "_stock", "_crop_logo", "_still", "_reuse"], "no photo-first, no category step"


def test_a_footage_shot_tries_video_then_photo_then_its_category_then_reuse(tmp_path):
    sh = _shot(0)
    sel, _ = _sel(tmp_path, [sh])
    order = []
    _record(sel, order, *ALL_STEPS)
    sel.select(sh)
    assert order == ["_youtube", "_stock", "_crop_logo", "_still", "_category", "_reuse"]


def test_a_photo_shot_tries_its_photo_first_and_youtube_for_one_round_only(tmp_path):
    sh = _shot(0, visual_kind="photo")
    sel, _ = _sel(tmp_path, [sh])
    order, rounds = [], []
    _record(sel, order, *ALL_STEPS)

    def youtube(shot, best, rounds=None, cap=None):
        order.append("_youtube")
        rounds_seen.append(rounds)
    rounds_seen = rounds
    sel._youtube = youtube
    sel.select(sh)
    assert order == ["_still", "_youtube", "_stock", "_crop_logo", "_category", "_reuse"], "the photo is not tried twice"
    assert rounds == [1]


def test_a_graphic_shot_is_filled_like_footage_while_graphics_are_off(tmp_path):
    sh = _shot(0, visual_kind="graphic", graphic="spec")
    sel, _ = _sel(tmp_path, [sh])
    order = []
    _record(sel, order, *ALL_STEPS)
    sel.select(sh)
    assert order[:2] == ["_youtube", "_stock"]


def test_a_found_photo_ends_the_ladder(tmp_path):
    sh = _shot(0, visual_kind="photo")
    sel, _ = _sel(tmp_path, [sh])
    order = []
    _record(sel, order, *ALL_STEPS)
    photo = Decision(shot=0, status="accepted", provider="still_commons", source_id="1", url="u")
    sel._still = lambda shot, only=None: order.append("_still") or photo
    assert sel.select(sh) is photo and order == ["_still"]


# --- the category step ----------------------------------------------------------------------------------
def _weak(rel):
    return Decision(shot=0, status="rejected", provider="youtube", source_id="v", url="u",
                    score=ClipScore(relevance=rel, total=50))


def _cat_case(tmp_path, monkeypatch, **kw):
    import ytauto.select as SEL
    sh = _shot(0)
    sel, proj = _sel(tmp_path, [sh], gem=object(), **kw)
    alt = sh.brief.model_copy(update={"subject": "submachine gun", "category_of": sh.brief.subject, "entities": []})
    monkeypatch.setattr(SEL, "generalise_brief", lambda gem, shot: alt)
    seen = []
    sel._youtube = lambda shot, best, rounds=None, cap=None: seen.append(("yt", shot.brief.subject, rounds, cap))
    sel._stock = lambda shot, best, cap=None: seen.append(("stock", shot.brief.subject, cap))
    return sh, sel, proj, seen


def test_the_category_is_searched_only_when_nothing_close_turned_up(tmp_path, monkeypatch):
    sh, sel, _, seen = _cat_case(tmp_path, monkeypatch)
    sel._category(sh, [_weak(30)])
    assert seen == [("yt", "submachine gun", 1, 4), ("stock", "submachine gun", 4)]
    seen.clear()
    sel._category(sh, [_weak(55)])
    assert seen == [], "a near miss of the real thing (relevance 55) beats a clip of its category"


@pytest.mark.parametrize("kw", [{"visual_ladder": False}, {"category_fallback": False}])
def test_the_category_step_can_be_switched_off(tmp_path, monkeypatch, kw):
    sh, sel, _, seen = _cat_case(tmp_path, monkeypatch, **kw)
    assert sel._category(sh, []) is None and seen == []


def test_generic_shots_and_category_briefs_are_not_rewritten_again(tmp_path, monkeypatch):
    sh, sel, _, seen = _cat_case(tmp_path, monkeypatch)
    sel._category(sh.model_copy(update={"brief": sh.brief.model_copy(update={"generic": True})}), [])
    sel._category(sh.model_copy(update={"brief": sh.brief.model_copy(update={"category_of": "x"})}), [])
    assert seen == []


def test_every_category_decision_is_marked_in_the_ledger(tmp_path):
    sh = _shot(0)
    sel, proj = _sel(tmp_path, [sh])
    alt = sh.model_copy(update={"brief": sh.brief.model_copy(update={"subject": "submachine gun",
                                                                         "category_of": "PPK-20 submachine gun"})})
    c = Candidate(provider="youtube", source_id="v", url="u")
    d = sel._decide(alt, c, "accepted", notes="window 3-7")
    assert d.notes == "window 3-7 | CATEGORY FALLBACK: submachine gun (for PPK-20 submachine gun)"
    assert "CATEGORY" not in sel._decide(sh, c, "rejected").notes, "an ordinary brief is not marked"


# --- the attempt budget ----------------------------------------------------------------------------------
class _Prov:
    name = "pexels"

    def __init__(self, n=30):
        self.n, self.fetched = n, 0

    def available(self):
        return True

    def search(self, q, k):
        return [Candidate(provider="pexels", source_id=f"{q}-{i}", url="u", height=1080) for i in range(self.n)]

    def fetch(self, c, folder):
        self.fetched += 1
        return Path("x.mp4")


def _stock_case(tmp_path, **kw):
    sh = _shot(0, stock_queries=["gun", "rifle"])
    prov = _Prov()
    sel, proj = _sel(tmp_path, [sh], providers={"pexels": prov}, stock_tries_per_provider=50, **kw)
    sel._dl_base[0] = 0

    def judge(shot, c, path, *a, **k):
        return sel._decide(shot, c, "rejected", score=ClipScore(relevance=40, total=40), rejects=["relevance 40 < 60"]), None
    sel._try_file = judge
    return sh, sel, prov


def test_stock_stops_at_the_shots_attempt_budget(tmp_path):
    sh, sel, prov = _stock_case(tmp_path, shot_attempts_budget=5)
    assert sel._stock(sh, []) is None and prov.fetched == 5
    sh, sel, prov = _stock_case(tmp_path / "cat", shot_attempts_budget=5)
    sel._stock(sh, [], cap=2)
    assert prov.fetched == 2, "the category step brings its own, smaller budget"


@pytest.mark.parametrize("kw", [{"visual_ladder": False}, {"shot_attempts_budget": 0}])
def test_the_attempt_budget_is_off_with_the_ladder_off_or_at_zero(tmp_path, kw):
    sh, sel, prov = _stock_case(tmp_path, **kw)
    sel._stock(sh, [])
    assert prov.fetched == 50, "only the provider's own tries limit, as before"


# --- disclosure ---------------------------------------------------------------------------------------
def test_the_verify_prompt_tells_the_judge_the_category_is_acceptable(tmp_path, monkeypatch):
    import ytauto.inspect.verify as V
    prompts = []

    class J:
        def json_video_file(self, clip, prompt, schema, **kw):
            prompts.append(prompt)
            return {"relevance": 80}

    monkeypatch.setattr(V, "needs_ai_check", lambda *a, **k: False, raising=False)
    sh = _shot(0)
    alt = sh.model_copy(update={"brief": sh.brief.model_copy(update={"subject": "submachine gun",
                                                                         "category_of": "PPK-20 submachine gun"})})
    for s in (sh, alt):
        try:
            V.verify_clip(J(), tmp_path / "c.mp4", s, "id", Settings())
        except Exception:                      # only the prompt matters here
            pass
    assert len(prompts) == 2
    assert "CATEGORY" not in prompts[0]
    assert 'exact "PPK-20 submachine gun"' in prompts[1] and "shows submachine gun" in prompts[1]


def test_the_report_lists_every_substitution_by_kind():
    rows = [{"shot": 1, "status": "accepted", "provider": "youtube", "notes": ""},
            {"shot": 2, "status": "accepted", "provider": "still_commons", "notes": ""},
            {"shot": 3, "status": "accepted", "provider": "youtube", "notes": "CATEGORY FALLBACK: smg (for PPK-20)"},
            {"shot": 4, "status": "accepted", "provider": "youtube", "notes": "REUSED footage of shot 1"},
            {"shot": 5, "status": "weak", "provider": "pexels", "notes": "WEAK MATCH: closest available"}]
    assert substitutions(rows) == {"photo": [2], "category": [3], "reuse": [4], "weak": [5]}


def test_the_review_is_told_a_category_shot_shows_its_category(tmp_path):
    from ytauto.review import _category_note
    sh = _shot(3)
    _, proj = _sel(tmp_path, [sh])
    assert _category_note(proj, sh) == ""
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    proj.log_decision(Decision(shot=3, status="accepted", provider="youtube", source_id="v", url="u", clip_path=str(clip),
                               notes="w | CATEGORY FALLBACK: submachine gun (for PPK-20 submachine gun)"))
    assert "a clear shot of submachine gun counts" in _category_note(proj, sh)


# --- photo-first searches Commons only; Commons is polite ---------------------------------------------
def test_photo_first_asks_only_the_configured_photo_providers(tmp_path):
    sh = _shot(0, visual_kind="photo")
    sel, _ = _sel(tmp_path, [sh])
    order, seen = [], []
    _record(sel, order, *ALL_STEPS)
    sel._still = lambda shot, only=None: seen.append(only)
    sel.select(sh)
    assert seen == [["commons"]], "Pexels/Pixabay photos of a named product are generic and would use up the tries"
    sel2, _ = _sel(tmp_path / "two", [_shot(1)])
    seen.clear()
    _record(sel2, [], *ALL_STEPS)
    sel2._still = lambda shot, only=None: seen.append(only)
    sel2.select(sel2.shots[0])
    assert seen == [None], "a normal shot's later photo step still tries every provider"


def test_the_photo_step_skips_providers_outside_the_filter(tmp_path):
    sh = _shot(0)
    sel, _ = _sel(tmp_path, [sh], gem=object())

    class P:
        def __init__(self, name):
            self.name, self.asked = name, 0

        def available(self):
            return True

        def search(self, q, n):
            self.asked += 1
            return []

    sel._photos = {"pexels": P("pexels"), "commons": P("commons")}
    sel._still(sh, only=["commons"])
    assert sel._photos["pexels"].asked == 0 and sel._photos["commons"].asked > 0
    sel._still(sh)
    assert sel._photos["pexels"].asked > 0


class _Resp:
    def __init__(self, status=200, headers=None, pages=()):
        self.status_code, self.headers, self._pages = status, headers or {}, list(pages)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code} error")

    def json(self):
        return {"query": {"pages": self._pages}}


def _page(i):
    return {"pageid": i, "index": i, "title": f"File:p{i}.jpg", "imageinfo": [{
        "mime": "image/jpeg", "width": 3000, "height": 2000, "url": f"u{i}", "descriptionurl": f"d{i}",
        "extmetadata": {"LicenseShortName": {"value": "CC0"}}}]}


@pytest.fixture
def commons(monkeypatch, tmp_path):
    """A Commons whose network and clock are fake; shared throttle state reset per test."""
    from ytauto.sources import images as IMG
    now = [1000.0]
    calls, sleeps = [], []
    monkeypatch.setattr(IMG.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(IMG.time, "sleep", lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)))
    monkeypatch.setattr(IMG.Commons, "_last", 0.0)
    monkeypatch.setattr(IMG.Commons, "_blocked_until", 0.0)
    responses = []

    def get(url, **kw):
        calls.append(kw)
        return responses.pop(0) if responses else _Resp(pages=[_page(1)])
    monkeypatch.setattr(IMG.requests, "get", get)
    c = IMG.Commons(Settings(commons_gap=1.0, contact="me@example.com"), Cache(tmp_path / "c"))
    return c, calls, sleeps, responses, now


def test_commons_calls_are_spaced_by_the_gap(commons):
    c, calls, sleeps, _, now = commons
    c.search("a", 4)
    c.search("b", 4)
    assert len(calls) == 2 and sleeps and abs(sleeps[0] - 1.0) < 1e-6, "the second call waited the gap"


def test_commons_sends_a_descriptive_user_agent(commons):
    c, calls, *_ = commons
    c.search("a", 4)
    assert "me@example.com" in calls[0]["headers"]["User-Agent"]


def test_a_429_pauses_commons_for_retry_after_and_is_not_cached(commons):
    c, calls, sleeps, responses, now = commons
    responses.append(_Resp(429, {"Retry-After": "30"}))
    assert c.search("a", 4) == []
    assert c.search("b", 4) == [] and len(calls) == 1, "paused: no second request"
    now[0] += 31
    got = c.search("a", 4)
    assert len(calls) == 2 and [x.source_id for x in got] == ["1"], "after the pause it asks again; the 429 was not cached"


def test_a_429_without_retry_after_pauses_for_a_minute(commons):
    c, calls, _, responses, now = commons
    responses.append(_Resp(429))
    c.search("a", 4)
    now[0] += 59
    assert c.search("b", 4) == [] and len(calls) == 1
    now[0] += 2
    c.search("b", 4)
    assert len(calls) == 2


def test_an_answered_query_is_cached_and_makes_no_second_request(commons):
    c, calls, *_ = commons
    first = c.search("ak-12", 4)
    again = c.search("ak-12", 4)
    assert len(calls) == 1 and [x.source_id for x in first] == [x.source_id for x in again] == ["1"]
    assert again[0].provider == "still_commons"


def test_a_failed_request_is_not_cached(commons):
    c, calls, _, responses, _ = commons
    responses.append(_Resp(500))
    assert c.search("a", 4) == []
    assert [x.source_id for x in c.search("a", 4)] == ["1"] and len(calls) == 2
