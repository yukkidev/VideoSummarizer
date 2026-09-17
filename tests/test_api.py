from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import api, store
from pipeline.config import Config
from pipeline.llm import ModelStatus
from pipeline.models import (
    Answer,
    Chunk,
    Citation,
    Segment,
    Summary,
    Transcript,
    Video,
)

INFO = {
    "id": "vid1",
    "title": "Video One",
    "uploader": "Uploader",
    "duration": 30,
    "description": "d",
    "webpage_url": "https://example.com/vid1",
    "thumbnail": None,
    "subtitles": {"en": [{}]},
    "automatic_captions": {},
}


class FakeLLM:
    def __init__(self):
        self.generate_calls = 0
        self.embed_calls = 0
        self.model = "test-model"

    def generate(self, prompt, **kwargs):
        self.generate_calls += 1
        return "answer text"

    def embed(self, texts, model=None, timeout=None):
        self.embed_calls += 1
        return [[0.1, 0.2, 0.3] for _ in texts]

    def refresh(self, model=None):
        return ModelStatus(server_ok=True, current=model or self.model)


@pytest.fixture()
def fake_llm(monkeypatch):
    llm = FakeLLM()

    def fake_client(cfg=None):
        return llm

    monkeypatch.setattr(api, "client_for", fake_client)
    return llm


@pytest.fixture()
def fake_downloads(monkeypatch, tmp_data: Path):
    calls = {"fetch": 0, "fetch_info": 0}

    def fake_fetch_info(url, **kwargs):
        calls["fetch_info"] += 1
        return dict(INFO)

    def fake_fetch(url, out_dir, plan=None, progress=None):
        calls["fetch"] += 1
        vdir = Path(out_dir) / "vid1"
        vdir.mkdir(parents=True, exist_ok=True)
        audio = vdir / "vid1.mp3"
        video = vdir / "vid1.mp4"
        subs = vdir / "vid1.en.vtt"
        if plan is None or plan.want_audio:
            audio.write_bytes(b"a")
        if plan is None or plan.want_video:
            video.write_bytes(b"v")
        if plan is None or plan.want_subtitles:
            subs.write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello world.\n"
            )
        video_obj = Video(
            video_id="vid1",
            url=INFO["webpage_url"],
            title=INFO["title"],
            author=INFO["uploader"],
            duration=30.0,
            audio_path=str(audio) if audio.exists() else None,
            video_path=str(video) if video.exists() else None,
            subtitle_path=str(subs) if subs.exists() else None,
        )
        store.save_video(out_dir, video_obj)
        return video_obj

    monkeypatch.setattr(api.downloader, "fetch_info", fake_fetch_info)
    monkeypatch.setattr(api.downloader, "fetch", fake_fetch)
    return calls


@pytest.fixture()
def fake_transcribe(monkeypatch):
    calls = {"n": 0}

    def fake(*, audio_path=None, subtitle_path=None, **kwargs):
        calls["n"] += 1
        return Transcript(
            text="Hello world. This is a test.",
            segments=[
                Segment(0.0, 1.0, "Hello world."),
                Segment(1.0, 2.0, "This is a test."),
            ],
            source="fake",
            language="en",
            duration=2.0,
        )

    monkeypatch.setattr(api, "transcribe", fake)
    return calls


@pytest.fixture()
def fake_summarize(monkeypatch):
    calls = {"n": 0}

    def fake(chunks, video, llm, model=None, progress=None):
        calls["n"] += 1
        return Summary(
            video_id=video.video_id,
            url=video.url,
            title=video.title,
            author=video.author,
            model=model or llm.model,
            summary="the summary",
            highlights=["q1", "q2"],
            created="2026-01-01T00:00:00+00:00",
        )

    monkeypatch.setattr(api, "summarize_chunks", fake)
    return calls


def test_process_full_flow(tmp_data: Path, fake_downloads, fake_transcribe, fake_summarize, fake_llm):
    record = api.process("https://example.com/vid1", out_dir=str(tmp_data))
    assert record["video_id"] == "vid1"
    assert record["title"] == "Video One"
    assert record["summary"]["summary"] == "the summary"
    assert record["transcript"]["source"] == "fake"
    assert fake_downloads["fetch"] == 1
    assert fake_transcribe["n"] == 1
    assert fake_summarize["n"] == 1
    vdir = tmp_data / "vid1"
    for name in ("metadata.json", "transcript.json", "chunks.json", "summary.json", "summary.md"):
        assert (vdir / name).exists(), name


def test_process_uses_cache_on_second_run(tmp_data: Path, fake_downloads, fake_transcribe, fake_summarize, fake_llm):
    api.process("https://example.com/vid1", out_dir=str(tmp_data))
    record = api.process("https://example.com/vid1", out_dir=str(tmp_data))
    assert record["summary"]["summary"] == "the summary"
    assert fake_downloads["fetch"] == 1
    assert fake_transcribe["n"] == 1
    assert fake_summarize["n"] == 1


def test_process_force_redoes_work(tmp_data: Path, fake_downloads, fake_transcribe, fake_summarize, fake_llm):
    api.process("https://example.com/vid1", out_dir=str(tmp_data))
    api.process("https://example.com/vid1", out_dir=str(tmp_data), force=True)
    assert fake_downloads["fetch"] == 2
    assert fake_transcribe["n"] == 2
    assert fake_summarize["n"] == 2


def test_process_subtitle_only(tmp_data: Path, fake_downloads, fake_transcribe, fake_summarize, fake_llm):
    record = api.process(
        "https://example.com/vid1",
        out_dir=str(tmp_data),
        want_audio=False,
        want_video=False,
        force_subtitles=True,
    )
    assert record["transcript"]["source"].startswith("subtitles")
    assert fake_transcribe["n"] == 0
    assert record["audio_path"] is None


def test_ask_builds_and_saves_vectors(tmp_data: Path, fake_downloads, fake_transcribe, fake_summarize, fake_llm, monkeypatch):
    api.process("https://example.com/vid1", out_dir=str(tmp_data))

    def fake_answer(video, chunks, question, llm, **kwargs):
        return Answer(
            video_id=video.video_id,
            question=question,
            answer="the answer",
            model="test-model",
            citations=[Citation(0, 0.0, 1.0, "Hello world.", 0.9)],
        )

    monkeypatch.setattr(api, "answer_question", fake_answer)
    payload = api.ask("vid1", "what is this?", out_dir=str(tmp_data))
    assert payload["answer"]["answer"] == "the answer"
    assert payload["answer"]["citations"][0]["start"] == 0.0
    assert (tmp_data / "vid1" / "vectors.json").exists()
    history = store.load_answers(tmp_data, "vid1")
    assert history and history[-1]["question"] == "what is this?"
    assert fake_llm.embed_calls >= 1


def test_ask_reuses_stored_vectors(tmp_data: Path, fake_downloads, fake_transcribe, fake_summarize, fake_llm, monkeypatch):
    api.process("https://example.com/vid1", out_dir=str(tmp_data))
    builds = {"n": 0}
    real_build = api.build_vectors

    def counting_build(chunks, llm, **kwargs):
        builds["n"] += 1
        return real_build(chunks, llm, **kwargs)

    monkeypatch.setattr(api, "build_vectors", counting_build)
    monkeypatch.setattr(api, "answer_question", lambda *a, **k: Answer(answer="x"))
    api.ask("vid1", "q1", out_dir=str(tmp_data))
    assert builds["n"] == 1
    api.ask("vid1", "q2", out_dir=str(tmp_data))
    assert builds["n"] == 1


def test_ask_missing_video(tmp_data: Path, fake_llm):
    with pytest.raises(Exception, match="no processed video"):
        api.ask("missing-id", "q", out_dir=str(tmp_data))


def test_ask_with_url_processes_first(tmp_data: Path, fake_llm, monkeypatch):
    def fake_process(ref, **kwargs):
        return {"video_id": "vid1"}

    monkeypatch.setattr(api, "process", fake_process)
    store.save_video(
        tmp_data,
        Video(video_id="vid1", url="u", title="T", author="A"),
    )
    store.save_chunks(tmp_data, "vid1", [Chunk(0, 0.0, 1.0, "text")])
    monkeypatch.setattr(api, "answer_question", lambda *a, **k: Answer(answer="x"))
    payload = api.ask("https://example.com/new", "q", out_dir=str(tmp_data))
    assert payload["answer"]["answer"] == "x"


def test_resolve_video_id(tmp_data: Path):
    store.save_video(tmp_data, Video(video_id="vid1", url="u", title="T", author="A"))
    cfg = Config(data_dir=str(tmp_data))
    assert api.resolve_video_id("vid1", cfg) == "vid1"
    assert api.resolve_video_id(str(tmp_data / "vid1"), cfg) == "vid1"
    assert api.resolve_video_id("nope", cfg) is None


def test_set_model_persists(tmp_data: Path, fake_llm):
    status = api.set_model("new-model", data_dir=str(tmp_data))
    assert status["server_ok"] is True
    assert Config.load(str(tmp_data)).model == "new-model"


def test_list_and_delete_videos(tmp_data: Path):
    store.save_video(tmp_data, Video(video_id="vid1", url="u", title="T", author="A"))
    videos = api.list_videos(out_dir=str(tmp_data))
    assert [v["video_id"] for v in videos] == ["vid1"]
    assert api.delete_video("vid1", out_dir=str(tmp_data)) is True
    assert api.delete_video("vid1", out_dir=str(tmp_data)) is False
