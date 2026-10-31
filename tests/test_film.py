"""film.json: facts are checked against the narration, cards are timed to the spoken words, styling is fixed."""
from ytauto import film as F
from ytauto.models import Segment

SEGS = [Segment(idx=0, start=0.0, end=8.0, text="A man dives deep on one breath."),
        Segment(idx=1, start=8.0, end=16.0, text="The country has 7,641 islands, and 37% of the world's coral species."),
        Segment(idx=2, start=16.0, end=26.0, text="In Palawan, an underground river runs 8.5 km under the mountains."),
        Segment(idx=3, start=26.0, end=40.0, text="Far to the north, farmers shaped the terraces of Ifugao by hand.")]
WORDS = [{"text": "The", "start": 8.1, "end": 8.3}, {"text": "country", "start": 8.3, "end": 8.7}, {"text": "has", "start": 8.7, "end": 8.9},
         {"text": "7,641", "start": 9.2, "end": 10.0}, {"text": "islands,", "start": 10.0, "end": 10.5},
         {"text": "and", "start": 10.5, "end": 10.7}, {"text": "37%", "start": 11.0, "end": 11.8},
         {"text": "In", "start": 16.1, "end": 16.2}, {"text": "Palawan,", "start": 16.2, "end": 16.8}]


def _raw():
    return {"tagline": "the story \"nobody\" told", "chapters": [0, 2, 3, 99],
            "stats": [{"segment": 1, "say": "7,641", "big": "7,641", "unit": "", "label": "islands", "major": True},
                      {"segment": 1, "say": "37%", "big": "37", "unit": "%", "label": "of the world's coral species"},
                      {"segment": 1, "say": "9,999", "big": "9,999", "label": "INVENTED"},          # not in the narration
                      {"segment": 2, "say": "8.5 km", "big": "8.5", "unit": " km", "label": "underground river"}],
            "places": [{"segment": 2, "name": "Palawan", "sub": "Puerto Princesa Underground River"},
                       {"segment": 3, "name": "Ifugao", "sub": "Rice terraces"},
                       {"segment": 3, "name": "Atlantis", "sub": "made up"}]}                       # not in the narration


def test_facts_the_narration_does_not_support_are_dropped():
    f = F.check_facts(_raw(), SEGS, "Philippines")
    assert [s["say"] for s in f["stats"]] == ["7,641", "37%", "8.5 km"]
    assert [p["name"] for p in f["places"]] == ["PALAWAN", "IFUGAO"]
    assert f["chapters"] == [2, 3], "sentence 0 and unknown sentences are not chapters"
    assert f["tagline"] == "the story nobody told"


def test_cards_appear_when_the_number_is_spoken_and_too_close_ones_are_dropped():
    stats, places, chapters = F.place_cards(F.check_facts(_raw(), SEGS, "Philippines"), SEGS, WORDS, 40.0)
    assert [(s["at"], s["side"]) for s in stats] == [(9.2, "left"), (16.0, "left")],         "7,641 at its word; 37% is 1.8 s later (dropped); 8.5 km has no matching word so it takes its sentence start"
    assert stats[0]["major"] is True and stats[0]["label"] == "ISLANDS" and stats[0]["to"] == 14.2
    assert [(p["name"], p["at"]) for p in places] == [("PALAWAN", 16.25), ("IFUGAO", 26.25)]
    assert chapters == [16.0], "26 s is under 20 s after 16 s"


def test_a_second_card_a_few_seconds_after_the_first_goes_on_the_other_side_and_a_third_is_dropped():
    segs = [Segment(idx=i, start=i * 4.0, end=i * 4.0 + 4.0, text=f"Fact {i + 1}00 here") for i in range(3)]
    facts = {"stats": [{"segment": i, "say": f"{i + 1}00", "big": f"{i + 1}00", "label": "x"} for i in range(3)]}
    stats, _, _ = F.place_cards(facts, segs, [], 60.0)
    assert [(s["at"], s["side"]) for s in stats] == [(0.0, "left"), (4.0, "right")], "the third would make three cards in 8 s"


def test_chapters_closer_than_twenty_seconds_are_merged():
    segs = [Segment(idx=i, start=i * 12.0, end=i * 12.0 + 12.0, text="x") for i in range(6)]
    _, _, chapters = F.place_cards({"chapters": [1, 2, 3, 5]}, segs, [], 80.0)
    assert chapters == [12.0, 36.0, 60.0]


def test_shots_get_fixed_cameras_and_a_chapter_opening_shot_gets_the_whip():
    raw = [{"id": f"s{i:02d}", "clip": f"s{i:02d}.mp4", "t0": i * 4.0, "t1": i * 4.0 + 4.0, "dur": 4.6} for i in range(10)]
    raw[5]["kind"] = "still"
    sh = F.style_shots(raw, [16.0])
    assert sh[0]["tr"] == "cut" and sh[4]["tr"] == "whip" and sh[4]["big"] is True
    assert sh[3]["tr"] == "zoom" and sh[1]["z"] == [1.12, 1.0]
    assert sh[5]["z"] == [1.0, 1.0] and sh[5]["px"] == [0, 0], "a still has its own move"
    assert F.style_shots(raw, [16.0]) == sh, "deterministic"


def test_build_assembles_title_end_card_and_sound_events():
    raw = [{"id": f"s{i:02d}", "clip": f"s{i:02d}.mp4", "t0": i * 8.0, "t1": i * 8.0 + 8.0, "dur": 8.6} for i in range(5)]
    f = F.build(slug="x", title="The Philippines", segments=SEGS, shots=raw, words=WORDS, seconds=40.0,
                facts=F.check_facts(_raw(), SEGS, "Philippines"), next_video="The reef", music="m.mp3")
    assert f["title"]["name"] == "THE PHILIPPINES" and f["title"]["at"] == 16.0 and f["title"]["tagline"] == "the story nobody told"
    assert f["end"] == {"at": 25.0, "kicker": "NEXT VIDEO", "title": "The reef"} and f["music"] == "m.mp3"
    kinds = [e["kind"] for e in f["events"]]
    assert kinds[0] == "hook" and "chapter" in kinds and "statbig" in kinds and kinds.count("impact") == 1
    none = F.build(slug="x", title="T", segments=SEGS, shots=raw, words=[], seconds=40.0, facts={}, show_title=False)
    assert none["title"] is None and none["end"] is None and none["chapters"] == [] and none["stats"] == []


def test_no_judge_means_no_facts_and_a_judge_failure_is_not_fatal():
    assert F.extract_facts(None, SEGS, "T", 40.0) == {}

    class Down:
        def json(self, *a, **k):
            raise RuntimeError("down")
    assert F.extract_facts(Down(), SEGS, "T", 40.0) == {}

    class Up:
        def json(self, prompt, schema, cache_key=None):
            assert "[2] (16-26s) In Palawan" in prompt
            return _raw()
    assert [p["name"] for p in F.extract_facts(Up(), SEGS, "Philippines", 40.0)["places"]] == ["PALAWAN", "IFUGAO"]


def test_place_names_are_spelled_as_the_narration_spells_them_and_invented_ones_dropped():
    segs = [Segment(idx=0, start=0, end=5, text="Taal volcano sits two days from Tubbataha Reef."),
            Segment(idx=1, start=5, end=10, text="Rice terraces.")]
    raw = {"places": [{"segment": 0, "name": "Tal", "sub": "volcano"}, {"segment": 0, "name": "Tubataha Reef", "sub": "reef"},
                      {"segment": 0, "name": "Atlantis", "sub": "?"}]}
    assert [p["name"] for p in F.check_facts(raw, segs, "Philippines")["places"]] == ["TAAL", "TUBBATAHA REEF"]


def test_no_place_chip_repeats_the_title_or_sits_on_the_title_card():
    raw = [{"id": f"s{i:02d}", "clip": f"s{i:02d}.mp4", "t0": i * 8.0, "t1": i * 8.0 + 8.0, "dur": 8.6} for i in range(5)]
    facts = F.check_facts(_raw(), SEGS, "Palawan Story")
    facts["places"].append({"segment": 2, "name": "PALAWAN STORY", "sub": ""})
    f = F.build(slug="x", title="Palawan Story", segments=SEGS, shots=raw, words=WORDS, seconds=40.0, facts=facts)
    assert [p["name"] for p in f["places"]] == ["IFUGAO"], "PALAWAN sits at 16.25 s, inside the title card (16.0-21.1 s)"


def test_sfx_audit_reads_the_plan_from_film_json(tmp_path):
    import json
    import pytest
    from ytauto import sfx
    plan = {"total": 10.0, "cues": [], "beds": [], "vo_mean_db": -20, "vo_peak_db": -3, "dropped": []}
    (tmp_path / "sfx.json").write_text(json.dumps(plan), encoding="utf-8")
    (tmp_path / "film.json").write_text(json.dumps({"sfx": plan, "shots": []}), encoding="utf-8")
    (tmp_path / "none.json").write_text(json.dumps({"sfx": None}), encoding="utf-8")
    assert sfx.load(tmp_path / "sfx.json") == plan == sfx.load(tmp_path / "film.json")
    with pytest.raises(ValueError):
        sfx.load(tmp_path / "none.json")


def test_a_title_or_place_in_another_script_still_works():
    segs = [Segment(idx=0, start=0, end=8, text="ঢাকা শহরে"), Segment(idx=1, start=8, end=20, text="সুন্দরবন বনে বাঘ আছে")]
    raw = [{"id": "s00", "clip": "a.mp4", "t0": 0.0, "t1": 8.0, "dur": 8.6}, {"id": "s01", "clip": "b.mp4", "t0": 8.0, "t1": 20.0, "dur": 12.6}]
    facts = F.check_facts({"places": [{"segment": 1, "name": "সুন্দরবন", "sub": "বন"}], "chapters": []}, segs, "বাংলাদেশ")
    f = F.build(slug="x", title="বাংলাদেশ", segments=segs, shots=raw, words=[], seconds=20.0, facts=facts)
    assert [p["name"] for p in f["places"]] == ["সুন্দরবন".upper()], "an empty Latin-only form of the title must not drop every place"


def test_a_film_is_refused_when_clips_are_missing_or_shots_leave_a_gap(tmp_path):
    for n in ("a.mp4", "b.mp4"):
        (tmp_path / n).write_bytes(b"x")
    rows = [{"id": "s00", "clip": "a.mp4", "t0": 0.0, "t1": 4.0}, {"id": "s01", "clip": "b.mp4", "t0": 4.0, "t1": 8.0}]
    assert F.check_complete(2, rows, tmp_path) == []
    assert "3 shots but shots.json has 2" in F.check_complete(3, rows, tmp_path)[0]
    (tmp_path / "b.mp4").unlink()
    assert F.check_complete(2, rows, tmp_path) == ["b.mp4 is missing from " + str(tmp_path)]
    gap = [rows[0], {"id": "s01", "clip": "a.mp4", "t0": 4.5, "t1": 8.0}]
    assert "s00 ends at 4.0s but s01 starts at 4.5s" in F.check_complete(2, gap, tmp_path)[0]
