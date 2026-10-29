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
