"""The pure parts of the measuring scripts (the rest needs your project folders, CLIP or your ChatGPT plan)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clip_gate_summary_finds_the_largest_gate_that_loses_nothing():
    g = _load("measure_clip_gate")
    # obvious misses (AI relevance <= 20) have low similarity, relevant clips (>= 60) high, a middle band in between
    items = [(0.10 + 0.01 * i, 5.0) for i in range(20)] + [(0.20 + 0.01 * i, 40.0) for i in range(10)] \
        + [(0.32 + 0.01 * i, 85.0) for i in range(10)]
    r = g.summarise(items)
    assert r["auc"] > 0.95 and r["pos"] == 10 and r["neg"] == 20
    b = r["best_safe"]
    assert b["lost_pos"] == 0 and b["caught_neg"] == 20, "the gate catches every miss and keeps every relevant clip"
    assert b["skipped"] >= 20


def test_clip_gate_reports_no_safe_gate_when_clip_cannot_separate_them():
    g = _load("measure_clip_gate")
    items = [(0.30, 5.0), (0.10, 90.0), (0.31, 5.0), (0.12, 85.0)]       # the relevant ones look LESS like the subject
    r = g.summarise(items)
    assert r["auc"] < 0.5
    assert r["best_safe"] is None or r["best_safe"]["caught_neg"] == 0


def test_clip_gate_finds_each_source_by_provider(tmp_path):
    g = _load("measure_clip_gate")
    (tmp_path / "sources" / "stock").mkdir(parents=True)
    (tmp_path / "sources" / "stills").mkdir(parents=True)
    (tmp_path / "sources" / "stock" / "pexels_77.mp4").write_bytes(b"x")
    (tmp_path / "sources" / "stills" / "still_pexels_9.jpg").write_bytes(b"x")
    mk = tmp_path / "mk"
    mk.mkdir()
    (mk / "2213-waterfall.mp4").write_bytes(b"x")
    assert g.find_source(tmp_path, {"provider": "pexels", "source_id": "77"}, mk).name == "pexels_77.mp4"
    assert g.find_source(tmp_path, {"provider": "still_pexels", "source_id": "9"}, mk).name == "still_pexels_9.jpg"
    assert g.find_source(tmp_path, {"provider": "mixkit", "source_id": "2213"}, mk).name == "2213-waterfall.mp4"
    assert g.find_source(tmp_path, {"provider": "coverr", "source_id": "1"}, mk) is None


def test_effort_comparison_counts_time_relevance_and_flag_disagreements():
    e = _load("measure_effort")
    ref = [{"relevance": 80, "secs": 6.0}, {"relevance": 30, "secs": 8.0, "watermark_or_logo": True}, {"relevance": 70, "secs": 7.0}]
    low = [{"relevance": 75, "secs": 3.0}, {"relevance": 35, "secs": 4.0}, {"relevance": 20, "secs": 3.5, "presenter_on_camera": True}]
    r = e.compare(ref, low)
    assert r["n"] == 3 and r["secs_ref"] == 7.0 and r["secs_other"] == 3.5
    assert r["relevance_within_tol"] == 2
    assert r["flag_disagreements"] == {"presenter_on_camera": 1, "watermark_or_logo": 1}
    assert r["clips_with_any_flag_disagreement"] == 2


def test_effort_disagreements_name_the_clip_the_flag_and_both_answers():
    e = _load("measure_effort")
    ref = [{"ai_or_animated": False}, {"ai_or_animated": True}, {"ai_or_animated": False}]
    other = [{"ai_or_animated": True}, {"ai_or_animated": True}, {"ai_or_animated": False}]
    assert e.disagreements(ref, other, [6, 19, 50]) == ["shot 6: ai_or_animated False -> True"]


def test_the_same_level_twice_keeps_two_separate_runs():
    """`--levels medium medium` is the noise floor; both runs must be stored apart (they used to share one key)."""
    src = (ROOT / "scripts" / "measure_effort.py").read_text(encoding="utf-8")
    assert "runs: list[list[dict]]" in src and "out: dict[str, list[dict]]" not in src


def test_photo_coverage_uses_the_photo_steps_queries_and_counts_covered_shots():
    m = _load("measure_photo_coverage")
    assert m.queries_for({"subject": "PPK-20", "setting": "Russia", "stock_queries": ["smg", "gun"]}) == \
        ["PPK-20 Russia", "PPK-20", "smg"]
    assert m.queries_for({"subject": "reef", "stock_queries": ["reef"]}) == ["reef"]
    hits = {1: {"commons": 3, "pexels": 0}, 2: {"commons": 0, "pexels": 0}, 3: {"commons": 0, "pexels": 2}}
    r = m.summarise(hits, {1: "photo", 2: "photo", 3: "footage"})
    assert r["all"]["any"] == (2, 3) and r["all"]["commons"] == (1, 3)
    assert r["photo"]["any"] == (1, 2) and r["footage"]["pexels"] == (1, 1)


def test_photo_coverage_counts_a_hit_as_exact_only_when_it_names_the_subjects_model():
    m = _load("measure_photo_coverage")
    ak = {"subject": "AK-12 assault rifle", "entities": ["Kalashnikov"]}
    assert m.distinctive(ak) == {"ak-12", "kalashnikov"}, "category words (rifle, assault) identify nothing"
    assert m.is_exact("Soldier with an AK-12 in a field", ak)
    assert m.is_exact("kalashnikov concern factory", ak)
    assert not m.is_exact("man holding rifle on range, russian soldier", ak), "a generic gun photo is not the AK-12"
    assert m.is_exact("AK12 rifle", {"subject": "AK-12"}) is False, "a different spelling is a miss, not a guess"
    assert not m.is_exact("", ak) and not m.is_exact("anything", {"subject": "rifle"}), "no distinctive word, nothing to match"
    assert m.is_exact("Lobaev SVL on a bench", {"subject": "Lobaev SVL rifle"})
