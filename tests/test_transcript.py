from ytauto.transcript import parse_text, split_shots


def test_range_lines():
    segs = parse_text("00:00–00:05: Intro to Japan\n00:05-00:10 Mount Fuji\n")
    assert [(s.start, s.end, s.text) for s in segs] == [(0, 5, "Intro to Japan"), (5, 10, "Mount Fuji")]


def test_single_timestamps_use_next_start():
    segs = parse_text("00:00 Hello\n00:07 World\n")
    assert segs[0].end == 7 and segs[1].end == 12


def test_srt_block():
    srt = "1\n00:00:00,000 --> 00:00:04,500\nHello there\n\n2\n00:00:04,500 --> 00:00:09,000\nSecond line\n"
    segs = parse_text(srt)
    assert len(segs) == 2 and segs[0].end == 4.5 and segs[1].text == "Second line"


def test_shot_split_bounds_and_coverage():
    segs = parse_text("00:00-00:05 a\n00:05-00:15 b\n00:15-00:17 c\n")
    shots = split_shots(segs)
    for s in shots:
        assert s.duration <= 6.0 + 1e-6
    assert shots[0].start == 0 and shots[-1].end == 17
    assert all(abs(a.end - b.start) < 1e-6 for a, b in zip(shots, shots[1:]))
    assert [s.segment for s in shots if s.segment == 1].__len__() == 2


# --- cut-off sentences (ElevenLabs breaks cues at 6 s) are joined ----------------------------
def _seg(i, a, b, t):
    from ytauto.models import Segment
    return Segment(idx=i, start=a, end=b, text=t)


def test_a_cut_sentence_is_joined_with_the_right_times_and_numbering():
    from ytauto.transcript import join_cut_sentences

    segs = [_seg(0, 0, 4, "It fires single shots, two-round bursts, or full"), _seg(1, 4, 4.3, "auto."),
            _seg(2, 5, 8, "The barrel has a chrome lining."), _seg(3, 9, 12, "A rail holds lights.")]
    out = join_cut_sentences(segs)
    assert [(s.idx, s.start, s.end, s.text) for s in out] == [
        (0, 0, 4.3, "It fires single shots, two-round bursts, or full auto."),
        (1, 5, 8, "The barrel has a chrome lining."), (2, 9, 12, "A rail holds lights.")]


def test_a_punctuated_transcript_without_cut_offs_is_unchanged():
    from ytauto.transcript import join_cut_sentences

    segs = [_seg(0, 0, 4, "One."), _seg(1, 4, 8, "Two?"), _seg(2, 8, 12, "Three!")]
    assert join_cut_sentences(segs) == segs


def test_lines_written_without_end_punctuation_are_never_joined():
    """Hand-written scripts (`00:00-00:05 Kyoto temples`) have no full stops: leave every line alone."""
    from ytauto.transcript import join_cut_sentences

    segs = [_seg(0, 0, 5, "Mount Fuji Japan"), _seg(1, 5, 10, "Kyoto temples"), _seg(2, 10, 15, "Tokyo at night."),
            _seg(3, 15, 20, "Osaka street food")]
    assert join_cut_sentences(segs) == segs


def test_one_missing_full_stop_cannot_swallow_a_whole_section():
    from ytauto.transcript import JOIN_MAX_SECONDS, join_cut_sentences

    run = [_seg(i, i * 5, i * 5 + 5, "and then more,") for i in range(8)]            # 40 s of commas
    segs = run + [_seg(8, 40, 45, "The end.")] + [_seg(9 + i, 45 + 5 * i, 50 + 5 * i, f"Line {i}.") for i in range(40)]
    out = join_cut_sentences(segs)
    assert max(s.end - s.start for s in out) <= JOIN_MAX_SECONDS, "a join never passes the cap"
    assert out[0].text.startswith("and then more, and then more,") and len(out[0].text) < len("and then more, " * 8)


def test_joined_cuts_leave_no_tiny_shots():
    from ytauto.transcript import join_cut_sentences, split_shots

    segs = [_seg(0, 0, 6, "The action fires single shots, two-round bursts, or full"), _seg(1, 6, 6.26, "auto."),
            _seg(2, 7, 11, "The barrel has a chrome lining."), _seg(3, 12, 15, "A rail holds lights.")]
    before = split_shots(segs)
    after = split_shots(join_cut_sentences(segs))
    assert min(s.end - s.start for s in before) < 0.5
    assert min(s.end - s.start for s in after) > 2.5


def test_json_transcripts_are_joined_too(tmp_path):
    import json as _json
    from ytauto.transcript import load_transcript

    p = tmp_path / "t.json"
    rows = [{"start": 0, "end": 4, "text": "Fires single shots or full"}, {"start": 4, "end": 4.3, "text": "auto."}]
    rows += [{"start": 5 + 4 * i, "end": 8 + 4 * i, "text": f"Line {i}."} for i in range(6)]
    p.write_text(_json.dumps(rows), encoding="utf-8")
    segs = load_transcript(p)
    assert segs[0].text == "Fires single shots or full auto." and (segs[0].start, segs[0].end) == (0, 4.3)
    assert len(segs) == 7
