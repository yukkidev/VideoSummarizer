from __future__ import annotations

from pathlib import Path

from pipeline import store
from pipeline.models import Answer, Chunk, Citation, Segment, Summary, Transcript, Video


def make_video(tmp_data: Path, *, with_files: bool = False) -> Video:
    vdir = tmp_data / "vid1"
    vdir.mkdir(parents=True, exist_ok=True)
    audio = video = subs = None
    if with_files:
        audio = str(vdir / "vid1.mp3")
        video = str(vdir / "vid1.mp4")
        subs = str(vdir / "vid1.en.vtt")
        for path in (audio, video, subs):
            Path(path).write_bytes(b"x")
    return Video(
        video_id="vid1",
        url="https://example.com/vid1",
        title="Title",
        author="Author",
        duration=90.0,
        audio_path=audio,
        video_path=video,
        subtitle_path=subs,
    )


def test_video_round_trip(tmp_data: Path):
    video = make_video(tmp_data)
    store.save_video(tmp_data, video)
    assert store.load_video(tmp_data, "vid1") == video
    assert store.load_video(tmp_data, "missing") is None


def test_transcript_round_trip(tmp_data: Path):
    transcript = Transcript(
        text="hello",
        segments=[Segment(0, 1, "hello")],
        source="test",
        language="en",
        duration=1.0,
    )
    store.save_transcript(tmp_data, "vid1", transcript)
    assert store.load_transcript(tmp_data, "vid1") == transcript


def test_chunks_round_trip(tmp_data: Path):
    chunks = [Chunk(0, 0.0, 1.0, "a"), Chunk(1, 1.0, 2.0, "b")]
    store.save_chunks(tmp_data, "vid1", chunks)
    assert store.load_chunks(tmp_data, "vid1") == chunks


def test_summary_round_trip(tmp_data: Path):
    summary = Summary(video_id="vid1", summary="s", highlights=["q"])
    store.save_summary(tmp_data, "vid1", summary)
    assert store.load_summary(tmp_data, "vid1") == summary
    store.save_summary_markdown(tmp_data, "vid1", "# hi")
    assert (tmp_data / "vid1" / "summary.md").read_text() == "# hi"


def test_vectors_round_trip(tmp_data: Path):
    store.save_vectors(tmp_data, "vid1", "nomic-embed-text", [[1.0, 2.0]], 1)
    loaded = store.load_vectors(tmp_data, "vid1")
    assert loaded["model"] == "nomic-embed-text"
    assert loaded["vectors"] == [[1.0, 2.0]]
    assert store.load_vectors(tmp_data, "nope") is None


def test_answers_append_and_load(tmp_data: Path):
    answer = Answer(
        video_id="vid1", question="q", answer="a", model="m",
        citations=[Citation(0, 0.0, 1.0, "text", 0.5)],
    )
    store.append_answer(tmp_data, answer, created="2026-01-01T00:00:00Z")
    store.append_answer(tmp_data, answer, created="2026-01-01T00:01:00Z")
    history = store.load_answers(tmp_data, "vid1")
    assert len(history) == 2
    assert history[0]["created"] == "2026-01-01T00:00:00Z"
    assert history[0]["citations"][0]["text"] == "text"
    assert store.load_answers(tmp_data, "nope") == []


def test_corrupt_files_are_tolerated(tmp_data: Path):
    (tmp_data / "vid1").mkdir(parents=True, exist_ok=True)
    (tmp_data / "vid1" / "metadata.json").write_text("{broken")
    (tmp_data / "vid1" / "chunks.json").write_text("not json")
    assert store.load_video(tmp_data, "vid1") is None
    assert store.load_chunks(tmp_data, "vid1") is None


def test_video_record_flags(tmp_data: Path):
    video = make_video(tmp_data, with_files=True)
    store.save_video(tmp_data, video)
    store.save_transcript(tmp_data, "vid1", Transcript(text="t", source="subs"))
    store.save_summary(tmp_data, "vid1", Summary(video_id="vid1", summary="s"))
    store.save_vectors(tmp_data, "vid1", "m", [[0.0]], 1)
    record = store.video_record(tmp_data, "vid1")
    assert record["has_video"] and record["has_audio"] and record["has_subtitles"]
    assert record["has_transcript"] and record["has_summary"] and record["has_vectors"]
    assert record["segment_count"] == 0
    assert store.video_record(tmp_data, "missing") is None


def test_list_videos(tmp_data: Path):
    store.save_video(tmp_data, make_video(tmp_data))
    (tmp_data / "notavideo").mkdir()
    records = store.list_videos(tmp_data)
    assert len(records) == 1
    assert records[0]["video_id"] == "vid1"
    assert "mtime" in records[0]


def test_delete_video(tmp_data: Path):
    store.save_video(tmp_data, make_video(tmp_data))
    assert store.delete_video(tmp_data, "vid1") is True
    assert store.delete_video(tmp_data, "vid1") is False
