"""High-level facade used by the CLI, TUI, web GUI, and agents."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pipeline import downloader, store
from pipeline.chunker import chunks_from_transcript
from pipeline.config import Config
from pipeline.llm import LLMError, OllamaClient, client_from_config
from pipeline.models import Answer, Transcript, Video
from pipeline.rag import answer_question, build_vectors, now
from pipeline.summarizer import summarize_chunks, summary_to_markdown
from pipeline.transcriber import (
    TranscriptionError,
    transcribe,
    transcript_from_captions,
)

ProgressFn = Callable[[str, float], None]

_config_lock = threading.Lock()


def load_config(data_dir: str | None = None) -> Config:
    return Config.load(data_dir)


def save_config(cfg: Config) -> None:
    with _config_lock:
        cfg.save()


def client_for(cfg: Config | None = None) -> OllamaClient:
    return client_from_config(cfg or Config.load())


def models_status(
    model: str | None = None, *, config: Config | None = None,
) -> dict[str, Any]:
    cfg = config or Config.load()
    llm = client_for(cfg)
    status = llm.refresh(model or cfg.model)
    return status.to_dict()


def set_model(model: str, *, data_dir: str | None = None) -> dict[str, Any]:
    cfg = Config.load(data_dir)
    cfg.model = model
    cfg.save()
    return models_status(model, config=cfg)


def _is_file(path: str | None) -> bool:
    return bool(path and Path(path).exists())


def _emit(progress: ProgressFn | None, message: str, fraction: float) -> None:
    if progress:
        progress(message, max(0.0, min(1.0, fraction)))


def ensure_video(
    url: str,
    *,
    cfg: Config,
    want_audio: bool = True,
    want_video: bool = True,
    want_subtitles: bool = True,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> Video:
    data_dir = Path(cfg.data_dir)
    video: Video | None = None
    cached = False

    if not force:
        info = downloader.fetch_info(url)
        video_id = str(info.get("id") or "")
        if video_id:
            video = store.load_video(data_dir, video_id)
            cached = video is not None

    if video is None:
        plan = downloader.DownloadPlan(
            want_audio=want_audio,
            want_video=want_video,
            want_subtitles=want_subtitles,
        )
        return downloader.fetch(
            url, data_dir, plan=plan,
            progress=lambda m: _emit(progress, m, 0.05),
        )

    missing_audio = want_audio and not _is_file(video.audio_path)
    missing_video = want_video and not _is_file(video.video_path)
    missing_subs = want_subtitles and not _is_file(video.subtitle_path)
    if cached and not (missing_audio or missing_video or missing_subs):
        _emit(progress, "using cached download", 0.05)
        return video

    plan = downloader.DownloadPlan(
        want_audio=missing_audio,
        want_video=missing_video,
        want_subtitles=missing_subs,
    )
    fresh = downloader.fetch(
        url, data_dir, plan=plan,
        progress=lambda m: _emit(progress, m, 0.05),
    )
    if not plan.want_audio:
        fresh.audio_path = video.audio_path
    if not plan.want_video:
        fresh.video_path = video.video_path
    if not plan.want_subtitles:
        fresh.subtitle_path = video.subtitle_path
    store.save_video(data_dir, fresh)
    return fresh


def ensure_transcript(
    video: Video,
    *,
    cfg: Config,
    force: bool = False,
    force_subtitles: bool = False,
    progress: ProgressFn | None = None,
) -> Transcript:
    data_dir = Path(cfg.data_dir)
    if not force:
        cached = store.load_transcript(data_dir, video.video_id)
        if cached is not None and cached.segments:
            _emit(progress, "using cached transcript", 0.35)
            return cached

    if video.subtitle_path and not _is_file(video.subtitle_path):
        video.subtitle_path = None
    if video.audio_path and not _is_file(video.audio_path):
        video.audio_path = None

    _emit(progress, "transcribing", 0.35)
    try:
        if force_subtitles and video.subtitle_path:
            transcript = transcript_from_captions(video.subtitle_path, cfg.language)
        else:
            transcript = transcribe(
                audio_path=video.audio_path,
                subtitle_path=video.subtitle_path,
                model=cfg.whisper_model,
                device=cfg.device,
                language=cfg.language,
                force_subtitles=force_subtitles,
                progress=lambda m, p: _emit(progress, m, 0.35 + 0.3 * p),
            )
    except (TranscriptionError, LLMError) as exc:
        raise TranscriptionError(str(exc)) from exc
    store.save_transcript(data_dir, video.video_id, transcript)
    return transcript


def process(
    url: str,
    *,
    out_dir: str | None = None,
    want_audio: bool = True,
    want_video: bool = True,
    want_subtitles: bool = True,
    force: bool = False,
    model: str | None = None,
    whisper_model: str | None = None,
    force_subtitles: bool = False,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    cfg = Config.load(out_dir)
    if model:
        cfg.model = model
    if whisper_model:
        cfg.whisper_model = whisper_model
    cfg.save()
    cfg.data_dir = str(out_dir or cfg.data_dir)

    video = ensure_video(
        url, cfg=cfg, want_audio=want_audio, want_video=want_video,
        want_subtitles=want_subtitles, force=force, progress=progress,
    )
    transcript = ensure_transcript(
        video, cfg=cfg, force=force, force_subtitles=force_subtitles,
        progress=progress,
    )

    chunks = chunks_from_transcript(transcript)
    store.save_chunks(cfg.data_dir, video.video_id, chunks)

    summary = None if force else store.load_summary(cfg.data_dir, video.video_id)
    if summary is None:
        _emit(progress, "summarizing", 0.7)
        llm = client_for(cfg)
        summary = summarize_chunks(
            chunks, video, llm, model=cfg.model,
            progress=lambda m, p: _emit(progress, m, 0.7 + 0.29 * p),
        )
        store.save_summary(cfg.data_dir, video.video_id, summary)
        store.save_summary_markdown(
            cfg.data_dir, video.video_id, summary_to_markdown(summary, chunks),
        )

    _emit(progress, "done", 1.0)
    record = store.video_record(cfg.data_dir, video.video_id) or {}
    record["transcript"] = transcript.to_dict()
    record["chunks"] = [c.to_dict() for c in chunks]
    record["summary"] = summary.to_dict()
    return record


def resolve_video_id(ref: str, cfg: Config) -> str | None:
    ref = (ref or "").strip()
    if not ref:
        return None
    candidate = Path(ref)
    if candidate.is_dir() and (candidate / store.METADATA).exists():
        return candidate.name
    if (Path(cfg.data_dir) / ref / store.METADATA).exists():
        return ref
    return None


def ask(
    ref: str,
    question: str,
    *,
    out_dir: str | None = None,
    top_k: int | None = None,
    model: str | None = None,
    force_vectors: bool = False,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    cfg = Config.load(out_dir)
    if model:
        cfg.model = model
        cfg.save()
    if top_k:
        cfg.top_k = top_k

    video_id = resolve_video_id(ref, cfg)
    if video_id is None:
        if ref.startswith(("http://", "https://")):
            record = process(ref, out_dir=out_dir, model=model, progress=progress)
            video_id = record["video_id"]
        else:
            raise LLMError(f"no processed video found for '{ref}'")
    data_dir = cfg.data_dir

    video = store.load_video(data_dir, video_id)
    chunks = store.load_chunks(data_dir, video_id) or []
    if video is None:
        raise LLMError(f"no processed video found for '{video_id}'")
    if not chunks:
        transcript = store.load_transcript(data_dir, video_id)
        if transcript is not None:
            chunks = chunks_from_transcript(transcript)
            store.save_chunks(data_dir, video_id, chunks)
    if not chunks:
        raise LLMError(f"'{video_id}' has no transcript; process it first")

    llm = client_for(cfg)
    vectors = None
    stored = store.load_vectors(data_dir, video_id)
    if (
        not force_vectors
        and stored
        and stored.get("model") == cfg.embed_model
        and len(stored.get("vectors", [])) == len(chunks)
    ):
        vectors = stored["vectors"]
    else:
        vectors = build_vectors(chunks, llm, embed_model=cfg.embed_model, progress=progress)
        store.save_vectors(
            data_dir, video_id, cfg.embed_model, vectors, len(chunks),
        )

    _emit(progress, "asking", 0.3)
    answer: Answer = answer_question(
        video, chunks, question, llm,
        vectors=vectors, top_k=cfg.top_k, model=cfg.model,
        embed_model=cfg.embed_model, progress=progress,
    )
    store.append_answer(data_dir, answer, created=now())
    record = store.video_record(data_dir, video_id) or {}
    record["data_dir"] = data_dir
    return {"answer": answer.to_dict(), "video": record}


def get_video_record(ref: str, *, out_dir: str | None = None) -> dict[str, Any]:
    cfg = Config.load(out_dir)
    video_id = resolve_video_id(ref, cfg)
    if video_id is None:
        raise LLMError(f"no processed video found for '{ref}'")
    record = store.video_record(cfg.data_dir, video_id)
    if record is None:
        raise LLMError(f"no processed video found for '{video_id}'")
    return record


def list_videos(*, out_dir: str | None = None) -> list[dict[str, Any]]:
    cfg = Config.load(out_dir)
    return store.list_videos(cfg.data_dir)


def delete_video(ref: str, *, out_dir: str | None = None) -> bool:
    cfg = Config.load(out_dir)
    video_id = resolve_video_id(ref, cfg)
    if video_id is None:
        return False
    return store.delete_video(cfg.data_dir, video_id)


def answer_history(video_id: str, *, out_dir: str | None = None) -> list[dict[str, Any]]:
    cfg = Config.load(out_dir)
    return store.load_answers(cfg.data_dir, video_id)
