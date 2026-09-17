from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from pipeline.models import Segment
from pipeline.transcriber import (
    TranscriptionError,
    clean_caption_text,
    faster_whisper_available,
    parse_caption_file,
    transcribe,
    transcribe_audio,
    transcript_from_captions,
    whisper_model_name,
)

VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.000
Hello and welcome

00:00:02.000 --> 00:00:04.000
Hello and welcome to the show

00:00:04.000 --> 00:00:06.000
to the show today
"""

SRT = """1
00:00:01,000 --> 00:00:03,500
First line of speech

2
00:00:03,500 --> 00:00:05,000
Second line of speech
"""


def test_clean_caption_text():
    assert clean_caption_text("<c>Hi</c>&nbsp;there &amp; you") == "Hi there & you"
    assert clean_caption_text("  spaced\u200b  out ") == "spaced out"


def test_parse_vtt_dedupes_rolling_captions(tmp_path: Path):
    p = tmp_path / "a.en.vtt"
    p.write_text(VTT)
    segments = parse_caption_file(p)
    assert segments == [
        Segment(0.0, 4.0, "Hello and welcome to the show"),
        Segment(4.0, 6.0, "to the show today"),
    ]


def test_parse_srt(tmp_path: Path):
    p = tmp_path / "a.srt"
    p.write_text(SRT)
    segments = parse_caption_file(p)
    assert segments[0] == Segment(1.0, 3.5, "First line of speech")
    assert segments[1].text == "Second line of speech"


def test_parse_ignores_metadata_and_blank(tmp_path: Path):
    p = tmp_path / "bad.vtt"
    p.write_text("WEBVTT\n\nNOTE something\n\n00:00:00.000 --> 00:00:01.000\n\n")
    assert parse_caption_file(p) == []


def test_transcript_from_captions_raises_without_cues(tmp_path: Path):
    p = tmp_path / "empty.vtt"
    p.write_text("WEBVTT\n")
    with pytest.raises(TranscriptionError, match="no caption cues"):
        transcript_from_captions(p)


def test_transcript_from_captions_language(tmp_path: Path):
    p = tmp_path / "a.vtt"
    p.write_text(VTT)
    t = transcript_from_captions(p, "fr")
    assert t.language == "fr"
    assert t.source.startswith("subtitles")
    assert t.duration == 6.0


def test_whisper_model_name_precedence(monkeypatch):
    monkeypatch.setenv("VS_WHISPER_MODEL", "env-model")
    assert whisper_model_name("explicit") == "explicit"
    assert whisper_model_name(None) == "env-model"
    monkeypatch.delenv("VS_WHISPER_MODEL")
    assert whisper_model_name(None) == "small"


class _FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class _FakeInfo:
    language = "en"
    duration = 3.0


class _FakeWhisperModel:
    def __init__(self, name, device="auto", compute_type=None):
        self.name = name
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, path, **kwargs):
        assert Path(path).exists()
        return iter(
            [_FakeSegment(0.0, 2.0, " Hello "), _FakeSegment(2.0, 3.0, " world ")]
        ), _FakeInfo()


@pytest.fixture()
def fake_faster_whisper(monkeypatch):
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = _FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return module


def test_transcribe_audio_with_fake_backend(fake_faster_whisper, tmp_path: Path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    seen = []
    t = transcribe_audio(audio, model="tiny", device="cpu", progress=lambda m, p: seen.append(p))
    assert t.text == "Hello world"
    assert t.segments[0] == Segment(0.0, 2.0, "Hello")
    assert t.source == "faster-whisper/tiny"
    assert seen[-1] == 1.0


def test_transcribe_audio_propagates_model_error(fake_faster_whisper, tmp_path: Path, monkeypatch):
    class Boom(_FakeWhisperModel):
        def transcribe(self, path, **kwargs):
            raise RuntimeError("model load failed")

    fake_faster_whisper.WhisperModel = Boom
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="model load failed"):
        transcribe_audio(audio, model="tiny", device="cpu")


def test_transcribe_requires_a_source():
    with pytest.raises(TranscriptionError, match="no audio or subtitle"):
        transcribe()


def test_transcribe_prefers_audio_when_whisper_available(tmp_path: Path, fake_faster_whisper):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    subs = tmp_path / "a.vtt"
    subs.write_text(VTT)
    t = transcribe(audio_path=audio, subtitle_path=subs, model="tiny", device="cpu")
    assert t.source.startswith("faster-whisper")


def test_transcribe_force_subtitles(tmp_path: Path, fake_faster_whisper):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    subs = tmp_path / "a.vtt"
    subs.write_text(VTT)
    t = transcribe(audio_path=audio, subtitle_path=subs, force_subtitles=True)
    assert t.source.startswith("subtitles")


def test_transcribe_falls_back_to_subs_when_whisper_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "pipeline.transcriber.faster_whisper_available", lambda: False,
    )
    subs = tmp_path / "a.vtt"
    subs.write_text(VTT)
    t = transcribe(audio_path=None, subtitle_path=subs)
    assert t.source.startswith("subtitles")


@pytest.mark.live
def test_live_faster_whisper_available():
    assert faster_whisper_available(), "install faster-whisper to run live transcription"
