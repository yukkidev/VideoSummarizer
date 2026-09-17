from __future__ import annotations

import pytest

from pipeline.models import (
    Answer,
    Chunk,
    Citation,
    Segment,
    Summary,
    Transcript,
    Video,
    format_timestamp,
)


def test_video_round_trip():
    v = Video(
        video_id="abc123",
        url="https://example.com/v",
        title="T",
        author="A",
        duration=12.5,
        description="d",
        audio_path="/a.mp3",
        video_path="/v.mp4",
        subtitle_path="/s.vtt",
    )
    assert Video.from_dict(v.to_dict()) == v


def test_video_from_ytdlp_info():
    info = {
        "id": "xyz",
        "title": "Hello",
        "uploader": "Chan",
        "duration": 61,
        "description": "desc",
        "webpage_url": "https://example.com/x",
        "thumbnail": "https://example.com/t.jpg",
    }
    v = Video.from_dict({**info, "url": info["webpage_url"]})
    assert v.video_id == "xyz"
    assert v.author == "Chan"
    assert v.duration == 61.0


def test_video_duration_none_and_garbage():
    assert Video.from_dict({"id": "a", "duration": None}).duration is None
    assert Video.from_dict({"id": "a", "duration": "NA"}).duration is None


def test_transcript_round_trip(sample_transcript: Transcript):
    assert Transcript.from_dict(sample_transcript.to_dict()) == sample_transcript


def test_chunk_and_summary_round_trip():
    c = Chunk(index=1, start=1.0, end=2.0, text="hi")
    assert Chunk.from_dict(c.to_dict()) == c
    s = Summary(
        video_id="v1", url="u", title="t", author="a", model="m",
        summary="sum", highlights=["q1", "q2"],
    )
    assert Summary.from_dict(s.to_dict()) == s


def test_answer_serialization():
    ans = Answer(
        video_id="v1",
        question="q",
        answer="a",
        model="m",
        citations=[Citation(index=0, start=5.0, end=10.0, text="t", score=0.9)],
    )
    d = ans.to_dict()
    assert d["citations"][0]["start"] == 5.0


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "00:00"),
        (59, "00:59"),
        (60, "01:00"),
        (3599, "59:59"),
        (3600, "1:00:00"),
        (3661, "1:01:01"),
        (-5, "00:00"),
        (None, "00:00"),
    ],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_segment_round_trip():
    seg = Segment(1.5, 3.5, "hello")
    assert Segment.from_dict(seg.to_dict()) == seg
