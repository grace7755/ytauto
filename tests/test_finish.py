"""Finishing pass: word alignment, caption chunks, the ASS file, and a real (tiny) ffmpeg render."""
import shutil
import subprocess

import pytest

from ytauto import finish as F
from ytauto.models import Segment

SEGS = [Segment(idx=0, start=0.0, end=2.0, text="Alaska has no roads."),
        Segment(idx=1, start=2.0, end=4.0, text="It's home.")]


def test_heard_times_are_used_and_spelling_comes_from_the_transcript():
    heard = [("alaska", 0.1, 0.5), ("has", 0.55, 0.7), ("no", 0.75, 0.9), ("roads", 0.95, 1.4),
             ("it's", 2.2, 2.5), ("home", 2.55, 3.0)]
    w = F.align_words(SEGS, heard)
    assert [x.text for x in w] == ["Alaska", "has", "no", "roads.", "It's", "home."]
    assert w[0].start == pytest.approx(0.1) and w[5].start == pytest.approx(2.55)


def test_missed_words_are_interpolated_inside_their_segment():
    heard = [("alaska", 0.1, 0.5), ("roads", 1.5, 1.9)]
    w = F.align_words(SEGS, heard)
    assert 0.5 <= w[1].start < w[2].start < 1.5, "has/no fall between the heard neighbours"
    assert all(2.0 <= x.start <= 4.0 for x in w[4:]), "segment 2 spread inside its own span"
    assert all(a.start <= b.start for a, b in zip(w, w[1:]))


def test_a_mishearing_never_drags_a_word_into_another_sentence():
    heard = [("home", 0.2, 0.4)]                     # "home" heard inside segment 1: wrong
    w = F.align_words(SEGS, heard)
    assert w[-1].start >= 2.0


def test_no_whisper_at_all_spreads_words_by_length():
    w = F.align_words(SEGS, [])
    assert len(w) == 6 and w[0].start == 0.0 and w[3].end <= 2.0 + 1e-6


def test_chunks_break_at_punctuation_and_size():
    words = [F.Word(t, i, i + 0.5) for i, t in enumerate("one two three four five six. seven".split())]
    got = [[x.text for x in c] for c in F.chunks(words, max_words=4)]
    assert got == [["one", "two", "three", "four"], ["five", "six."], ["seven"]]


def test_ass_highlights_one_word_per_event_and_includes_cards():
    words = [F.Word("Alaska", 0.0, 0.5), F.Word("roads.", 0.5, 1.0)]
    ass = F.build_ass(words, [F.TextCard("QUINHAGAK, ALASKA", 1.0, 4.0)], "Arial")
    caps = [l for l in ass.splitlines() if ",Caption," in l]
    assert len(caps) == 2 and caps[0].count("&H0035C8FF&") == 1
    assert ",Lower," in ass and "QUINHAGAK" in ass
    assert "{" not in F._esc("a{b}")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_a_real_render_has_voice_music_captions_and_the_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "heard_words", lambda vo: [])
    video, vo, music, sfx = (tmp_path / n for n in ("v.mp4", "vo.wav", "m.wav", "s.wav"))
    run = lambda *a: subprocess.run(["ffmpeg", "-y", "-v", "error", *a], check=True)
    run("-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=4", "-pix_fmt", "yuv420p", str(video))
    run("-f", "lavfi", "-i", "sine=frequency=300:duration=4", str(vo))
    run("-f", "lavfi", "-i", "sine=frequency=500:duration=10", str(music))
    run("-f", "lavfi", "-i", "sine=frequency=900:duration=0.3", str(sfx))
    out = tmp_path / "out.mp4"
    spec = F.FinishSpec(vo=vo, music=music, sfx=[F.Sfx(sfx, 1.0)], grain=video, tail=1.0,
                        cards=[F.TextCard("TITLE", 0.2, 2.0, "title")])
    F.render(video, spec, SEGS, out, tmp_path / "work")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
                            "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert "video" in probe and "audio" in probe
    dur = float([l for l in probe.splitlines() if l.replace(".", "").isdigit()][0])
    assert 4.8 <= dur <= 5.2, "voice length + 1 s tail"
    assert (tmp_path / "work" / "captions.ass").exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_audio_is_decoded_by_ffmpeg_not_pyav(tmp_path):
    """faster-whisper's own decoder crashes on current PyAV (`metadata_errors`), which silently
    disabled `--vo` alignment; ffmpeg decodes it instead."""
    from ytauto.align import load_audio

    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    str(wav)], check=True)
    a = load_audio(wav)
    assert a.dtype.name == "float32" and 31000 <= len(a) <= 33000 and abs(a).max() > 0.05
