"""Planner rules measured against the 12-shot validation run of 2026-10-29.

That run produced five pairs of byte-identical briefs (one per segment, shared by both shots split
from it), plus two briefs no single clip can satisfy ("a drone shot transitioning from a temple to
a city crossing" and "a montage of quick cuts"). The first wasted about half the API budget; the
second burned 24 candidates and dropped the shot to stock. These tests encode those findings.
"""
import pytest

from ytauto.config import Settings
from ytauto.models import Brief, Segment, Shot
from ytauto.planner import PLAN_PROMPT, _differentiate, _fallback_brief, plan
from ytauto.transcript import split_shots

S = Settings()

# A 10 s segment splits into two shots; a 5 s one stays single.
SEGS = [Segment(idx=0, start=0.0, end=10.0, text="Number one: Mount Fuji, Japan's snow-capped volcano."),
        Segment(idx=1, start=10.0, end=15.0, text="Number two: Kyoto, home to thousands of temples.")]


def _shots():
    return split_shots(SEGS, S.shot_target, S.shot_min, S.shot_max)


class FakeJudge:
    """Returns whatever `payload` says, and records the prompt it was given."""

    def __init__(self, payload):
        self.payload, self.prompt = payload, None

    def json(self, prompt, schema, **kw):
        self.prompt = self.prompt or prompt      # the plan prompt; a review call may follow
        return self.payload


def test_split_shots_of_one_segment_get_different_briefs_offline():
    """The regression that mattered: two shots from one segment must not share a `visual`."""
    shots = _shots()
    same_seg = [sh for sh in shots if sh.segment == 0]
    assert len(same_seg) >= 2, "a 10s segment must split into at least two shots"

    briefs = plan(SEGS, shots, "Japan", S, None, None)        # gem=None -> heuristic path
    visuals = [briefs[sh.id].visual for sh in same_seg]
    assert len(set(visuals)) == len(visuals), f"identical briefs for one segment: {visuals}"


def test_briefs_are_keyed_by_shot_id_not_segment_idx():
    shots = _shots()
    briefs = plan(SEGS, shots, "Japan", S, None, None)
    assert set(briefs) == {sh.id for sh in shots}


def test_every_shot_gets_a_brief_with_queries():
    shots = _shots()
    briefs = plan(SEGS, shots, "Japan", S, None, None)
    for sh in shots:
        assert briefs[sh.id].youtube_queries, f"shot {sh.id} has no queries"
        assert briefs[sh.id].visual.strip()


def test_fallback_brief_varies_with_variant():
    shots = _shots()
    a, b = shots[0], shots[1]
    assert a.segment == b.segment and a.variant != b.variant
    ba = _fallback_brief(a, SEGS[0], "Japan", S.queries_per_segment)
    bb = _fallback_brief(b, SEGS[0], "Japan", S.queries_per_segment)
    assert ba.visual != bb.visual
    assert ba.youtube_queries != bb.youtube_queries, "query order must rotate per variant"
    assert ba.entities == bb.entities, "the subject must stay the same - only the framing changes"


def test_duplicate_visuals_from_the_model_are_repaired():
    """The model is told not to repeat itself but sometimes does; we must not pass that through."""
    shots = _shots()
    dup = "A breathtaking sunrise over Mount Fuji."
    briefs = {sh.id: Brief(visual=dup, youtube_queries=["q"]) for sh in shots if sh.segment == 0}
    repaired = _differentiate([sh for sh in shots if sh.segment == 0], briefs)

    assert repaired, "the repair must report which shots it touched"
    visuals = [b.visual for b in briefs.values()]
    assert len(set(visuals)) == len(visuals), f"still duplicated: {visuals}"
    assert any(v == dup for v in visuals), "the first shot keeps the original brief"


def test_differentiate_leaves_distinct_briefs_alone():
    shots = _shots()
    briefs = {sh.id: Brief(visual=f"view {sh.id}", youtube_queries=["q"]) for sh in shots}
    before = {k: v.visual for k, v in briefs.items()}
    assert _differentiate(shots, briefs) == []
    assert {k: v.visual for k, v in briefs.items()} == before


def test_same_visual_in_different_segments_is_not_touched():
    """Only shots *within* one segment are a pacing problem; across segments it is legitimate."""
    shots = _shots()
    one_per_seg = [next(sh for sh in shots if sh.segment == i) for i in (0, 1)]
    briefs = {sh.id: Brief(visual="same text", youtube_queries=["q"]) for sh in one_per_seg}
    assert _differentiate(one_per_seg, briefs) == []


def test_ai_briefs_are_used_and_keyed_by_shot_id():
    shots = _shots()
    payload = {"shots": [{"id": sh.id, "visual": f"planned shot {sh.id}", "entities": ["Mount Fuji"],
                          "youtube_queries": [f"mount fuji q{sh.id}"], "stock_queries": ["fuji"]} for sh in shots]}
    gem = FakeJudge(payload)
    briefs = plan(SEGS, shots, "Japan", S, None, gem)
    for sh in shots:
        assert briefs[sh.id].visual == f"planned shot {sh.id}"
        assert briefs[sh.id].youtube_queries == [f"mount fuji q{sh.id}"]


def test_ai_failure_falls_back_to_heuristic_briefs():
    class Boom:
        def json(self, *a, **kw):
            raise RuntimeError("all judge providers failed")

    shots = _shots()
    briefs = plan(SEGS, shots, "Japan", S, None, Boom())
    assert set(briefs) == {sh.id for sh in shots}
    assert all(briefs[sh.id].youtube_queries for sh in shots)


def test_partial_ai_response_is_topped_up_per_shot():
    """A model that answers for only some shots must not leave the rest brief-less."""
    shots = _shots()
    payload = {"shots": [{"id": shots[0].id, "visual": "planned", "entities": [],
                          "youtube_queries": ["q"], "stock_queries": []}]}
    briefs = plan(SEGS, shots, "Japan", S, None, FakeJudge(payload))
    assert set(briefs) == {sh.id for sh in shots}
    assert briefs[shots[0].id].visual == "planned"
    for sh in shots[1:]:
        assert briefs[sh.id].youtube_queries


def test_ai_brief_with_empty_queries_is_replaced():
    shots = _shots()
    payload = {"shots": [{"id": sh.id, "visual": "planned", "entities": [],
                          "youtube_queries": [], "stock_queries": []} for sh in shots]}
    briefs = plan(SEGS, shots, "Japan", S, None, FakeJudge(payload))
    assert all(briefs[sh.id].youtube_queries for sh in shots)


def test_prompt_forbids_the_two_impossible_brief_shapes():
    """Shot 0 of the validation run asked for a transition between two landmarks 6 km apart and
    shot 11 for a montage; `inspect/scenes.py` extracts ONE cut-free window, so neither can pass."""
    shots = _shots()
    gem = FakeJudge({"shots": []})
    plan(SEGS, shots, "Japan", S, None, gem)
    p = gem.prompt.lower()
    assert "montage" in p and "quick cuts" in p
    assert "one continuous shot" in p
    assert "one place" in p


def test_prompt_lists_shots_with_their_variant_and_segment():
    shots = _shots()
    gem = FakeJudge({"shots": []})
    plan(SEGS, shots, "Japan", S, None, gem)
    for sh in shots:
        assert f"[{sh.id}] segment {sh.segment}" in gem.prompt


@pytest.mark.parametrize("field", ["{title}", "{nq}", "{shots}", "{dur}"])
def test_prompt_template_placeholders_are_all_filled(field):
    shots = _shots()
    gem = FakeJudge({"shots": []})
    plan(SEGS, shots, "Japan", S, None, gem)
    assert field in PLAN_PROMPT and field not in gem.prompt


# --- impossible briefs -------------------------------------------------------
# Measured 2026-10-29: the planner produced "a drone shot that transitions from a temple garden to
# a futuristic cityscape" (two places 6 km apart) and "a montage of quick cuts". Neither can be
# filled, because inspect/scenes.py extracts a single cut-free window. The first cost 24 rejected
# candidates and a drop to stock; a live call still produced a montage brief after the prompt
# forbade it, so the repair has to be in code, not only in the prompt.

IMPOSSIBLE = [
    "A montage of quick cuts showing the most striking moments.",
    "A series of shots showing temples, food and mountains.",
    "A sweeping drone shot that transitions from an ancient temple garden to a futuristic cityscape.",
    "A shot of the temple that cuts to the city crossing.",
    "Various shots of Japanese landscapes.",
    "A split-screen of summer and winter in Hokkaido.",
    "A montage of quick, exciting clips ending with a question mark graphic.",
    "A compilation of the best views in Kyoto.",
]

# These are all ONE continuous take and must survive untouched - a camera that moves is still a
# single shot, and rejecting these would throw away the planner's best briefs.
POSSIBLE = [
    "A wide, cinematic drone shot capturing Mount Fuji at sunrise reflecting in a calm lake.",
    "A slow-motion walking tour shot through the Arashiyama Bamboo Grove.",
    "A dynamic, wide-angle drone shot looking down at the Shibuya Scramble Crossing at night.",
    "A close-up, steady shot of takoyaki being cooked, with steam rising.",
    "A slow pan across vast lavender fields in Furano.",
    "A low-angle shot of neon signs as people pass by.",
    "A drone shot gliding over a futuristic cityscape with temples visible in the distance.",
    "A static locked-off view of a traditional wooden street in Gion.",
]


@pytest.mark.parametrize("visual", IMPOSSIBLE)
def test_impossible_briefs_are_detected(visual):
    from ytauto.planner import _impossible_reason

    assert _impossible_reason(visual) is not None, f"should be rejected: {visual}"


@pytest.mark.parametrize("visual", POSSIBLE)
def test_single_take_camera_moves_are_kept(visual):
    """A moving camera is still one continuous shot; these must not be flagged."""
    from ytauto.planner import _impossible_reason

    assert _impossible_reason(visual) is None, f"should be allowed: {visual}"


def test_impossible_brief_is_replaced_but_research_is_kept():
    shots = _shots()
    sh = shots[0]
    briefs = {sh.id: Brief(visual="A montage of quick cuts of Japan.", entities=["Mount Fuji"],
                           youtube_queries=["mount fuji 4k"], stock_queries=["fuji"])}
    from ytauto.planner import _reject_impossible

    replaced = _reject_impossible([sh], briefs, {0: SEGS[0]}, "Japan", S.queries_per_segment)
    assert replaced, "the replacement must be reported"
    assert briefs[sh.id].visual.strip()
    assert "montage" not in briefs[sh.id].visual.lower()
    # the planner's research is still worth keeping
    assert briefs[sh.id].entities == ["Mount Fuji"]
    assert briefs[sh.id].youtube_queries == ["mount fuji 4k"]


def test_plan_end_to_end_never_emits_an_impossible_brief():
    """Whatever the model returns, `plan` must not hand one of these to the selector."""
    from ytauto.planner import _impossible_reason

    shots = _shots()
    payload = {"shots": [{"id": sh.id, "visual": "A montage of quick cuts, transitioning to a map graphic.",
                          "entities": ["Kyoto"], "youtube_queries": ["kyoto 4k"],
                          "stock_queries": ["kyoto"]} for sh in shots]}
    briefs = plan(SEGS, shots, "Japan", S, None, FakeJudge(payload))
    for sh in shots:
        assert _impossible_reason(briefs[sh.id].visual) is None, briefs[sh.id].visual


def test_replacement_uses_the_planners_entities_as_the_subject():
    """A narration-only beat ("Which would you visit first?") has no entity in its text, so the
    heuristic would name the video title - not a filmable subject. The planner's own entities are
    better, and they are still trustworthy when only the shot description was impossible."""
    from ytauto.planner import _reject_impossible

    seg = Segment(idx=0, start=0.0, end=5.0, text="Which of these would you visit first?")
    sh = Shot(id=0, segment=0, start=0.0, end=5.0, text=seg.text, variant=0)
    briefs = {0: Brief(visual="A montage of quick clips ending with a graphic.",
                       entities=["Mount Fuji", "Kyoto"], youtube_queries=["japan 4k"])}
    _reject_impossible([sh], briefs, {0: seg}, "My Video Title", S.queries_per_segment)

    v = briefs[0].visual
    assert "My Video Title" not in v, f"must not search for the video's own title: {v}"
    assert "Mount Fuji" in v



# --- subject-aimed queries and the video topic --------------------------------
# Measured on the 12-shot run of 2026-10-29: a takoyaki close-up had entities ["Dotonbori", "Osaka",
# "Takoyaki"] and three of four queries about the Dotonbori canal; YouTube returned canal walking
# tours, all 8 were rejected, and the shot fell to stock. The closing beat searched for the video's
# own title, and the impossible-brief repair rebuilt a two-place brief by joining two entities.

def test_primary_subject_is_the_entity_named_first_in_the_visual():
    from ytauto.planner import primary_subject

    b = Brief(visual="A close-up of takoyaki being cooked on a stall in Dotonbori, Osaka.",
              entities=["Dotonbori", "Osaka", "Takoyaki"])
    assert primary_subject(b) == "Takoyaki"


def test_queries_about_the_area_are_reaimed_at_the_subject():
    from ytauto.planner import _align_queries

    sh = Shot(id=0, segment=0, start=0, end=5, text="t")
    briefs = {0: Brief(visual="A close-up of takoyaki on a street stall in Dotonbori.",
                       entities=["Dotonbori", "Takoyaki"],
                       youtube_queries=["Osaka Dotonbori street food 4K", "Takoyaki cooking close up",
                                        "Dotonbori canal night vibrant", "Osaka food stall cinematic"])}
    assert _align_queries([sh], briefs, 4)
    qs = briefs[0].youtube_queries
    assert sum("takoyaki" in q.lower() for q in qs) >= 2
    assert "takoyaki" in qs[0].lower(), "subject queries go first: they are screened first"


def test_queries_already_on_subject_are_left_alone():
    """A multi-word name matches on any significant word ("bamboo" for Arashiyama Bamboo Grove);
    requiring every word re-aimed shots that had succeeded."""
    from ytauto.planner import _align_queries

    sh = Shot(id=0, segment=0, start=0, end=5, text="t")
    qs = ["Arashiyama Bamboo Grove drone 4K", "Kyoto bamboo forest cinematic",
          "Walking through bamboo forest Kyoto", "Peaceful Japan nature b-roll"]
    briefs = {0: Brief(visual="A drone shot over the Arashiyama Bamboo Grove.",
                       entities=["Arashiyama Bamboo Grove", "Kyoto"], youtube_queries=list(qs))}
    assert _align_queries([sh], briefs, 4) == []
    assert briefs[0].youtube_queries == qs


@pytest.mark.parametrize("title,topic", [
    ("5 Best Destinations in Japan - Full Validation", "Japan"),
    ("Top 10 Things to Do in Kyoto", "Kyoto"),
    ("Why Iceland Is Unreal", "Iceland"),
    ("Mount Fuji Travel Guide", "Mount Fuji"),
])
def test_topic_from_title(title, topic):
    from ytauto.planner import topic_from_title

    assert topic_from_title(title) == topic


def test_a_beat_that_names_nothing_never_searches_for_the_title():
    seg = Segment(idx=0, start=0.0, end=5.0, text="Which of these would you visit first?")
    sh = Shot(id=0, segment=0, start=0.0, end=5.0, text=seg.text)
    title = "5 Best Destinations in Japan - Full Validation"
    b = _fallback_brief(sh, seg, title, 4)
    assert all("Validation" not in q and "Destinations" not in q for q in b.youtube_queries), b.youtube_queries
    assert "Japan" in b.visual


def test_impossible_brief_repair_uses_one_subject_not_two_places():
    from ytauto.planner import _reject_impossible

    sh = Shot(id=0, segment=0, start=0, end=5, text="t")
    briefs = {0: Brief(visual="A drone shot that transitions from a Japanese garden to a Tokyo street.",
                       entities=["Japanese garden", "Tokyo street"], youtube_queries=["q"])}
    _reject_impossible([sh], briefs, {0: Segment(idx=0, start=0, end=5, text="t")}, "Japan", 4)
    v = briefs[0].visual
    assert not ("garden" in v.lower() and "street" in v.lower()), f"still two places: {v}"


def test_ai_topic_is_used_for_beats_that_name_nothing():
    seg = Segment(idx=0, start=0.0, end=5.0, text="Which of these would you visit first?")
    sh = Shot(id=0, segment=0, start=0.0, end=5.0, text=seg.text)
    gem = FakeJudge({"topic": "Iceland", "shots": []})      # no usable brief for the shot
    briefs = plan([seg], [sh], "My Channel Video", S, None, gem)
    assert "Iceland" in briefs[0].visual


def test_the_setting_is_pinned_on_every_brief_and_placeless_queries():
    from ytauto.models import Brief
    from ytauto.planner import with_setting

    b = with_setting(Brief(subject="Remote house", entities=["House", "village"],
                           youtube_queries=["House 4k", "remote Alaskan village 4K"]),
                     "Alaska", "This is not an emergency.")
    assert b.setting == "Alaska"
    assert b.youtube_queries == ["Alaska House 4k", "remote Alaskan village 4K"], "Alaskan already names it"
    assert "Alaska" in b.names() and "not usable" in b.names()
    assert with_setting(Brief(subject="x", youtube_queries=["q"]), "").youtube_queries == ["q"]


def test_cache_keys_of_briefs_without_a_setting_do_not_change():
    from ytauto.models import Brief

    assert Brief(entities=["a", "b"]).key() == "a,b"
    assert Brief(entities=["a"], setting="Alaska").key() == "a|Alaska"
    assert Brief(entities=["a", "b"]).names() == "a, b"


def test_queries_naming_a_place_from_the_narration_are_kept():
    from ytauto.models import Brief
    from ytauto.planner import with_setting

    b = with_setting(Brief(entities=["Hakuba"], youtube_queries=["hakuba ski 4k", "powder snow 4k"]),
                     "Japan", "Hakuba has some of the best powder snow.")
    assert b.youtube_queries == ["hakuba ski 4k", "Japan powder snow 4k"]


def test_a_config_brief_override_replaces_the_planned_brief(tmp_path, monkeypatch):
    import ytauto.pipeline as P
    from ytauto.config import Settings

    t = tmp_path / "t.txt"
    t.write_text("00:00-00:04 One.\n00:04-00:08 Two.", encoding="utf-8")
    s = Settings(projects_dir=tmp_path / "p", cache_dir=tmp_path / "c", judge_chain=[],
                 brief_overrides={1: {"subject": "cloudy sky", "youtube_queries": ["sky 4k"]}})
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("HEART_CONFIG", "")
    p = P.Pipeline(s, t, "Test")
    p.prepare()
    assert p.shots[1].brief.subject == "cloudy sky" and p.shots[1].brief.youtube_queries == ["sky 4k"]
    assert p.shots[0].brief.subject != "cloudy sky"


def test_presenter_words_are_stripped_from_youtube_queries():
    """5-guns film: most rejects were review channels with a presenter; "review" in the query found them."""
    from ytauto.models import Brief
    from ytauto.planner import _b_roll_queries

    b = Brief(visual="v", subject="Glock 19", youtube_queries=[
        "Glock 19 review", "Glock 19 vs Sig P365 comparison", "Glock 19 slow motion 4k",
        "Glock 19 b-roll review", "Talkeetna aerial"])
    briefs = {0: b}
    fixed = _b_roll_queries(briefs)
    assert briefs[0].youtube_queries == ["Glock 19 b-roll", "Glock 19 Sig P365 b-roll",
                                         "Glock 19 slow motion 4k", "Talkeetna aerial"]
    assert len(fixed) == 3
    clean = {1: Brief(visual="v", youtube_queries=["Kyoto temples 4k"])}
    assert _b_roll_queries(clean) == [] and clean[1].youtube_queries == ["Kyoto temples 4k"]
