"""High-level facade used by the CLI, TUI, web GUI, and agents."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pipeline import downloader, store
from pipeline.chunker import chunks_from_transcript
from pipeline.config import Config
from pipeline.llm import LLMError, OllamaClient, client_from_config
from pipeline.models import (
    Answer,
    ChatMessage,
    ChatThread,
    Chunk,
    Citation,
    Playlist,
    Transcript,
    Video,
    estimate_tokens,
)
from pipeline.rag import (
    answer_question,
    build_vectors,
    generate_reply_detailed,
    now,
    search,
    search_all,
)
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


def _ensure_vectors(
    cfg: Config,
    llm: OllamaClient,
    video_id: str,
    chunks: list[Chunk],
    *,
    progress: ProgressFn | None = None,
    force: bool = False,
) -> list[list[float]]:
    stored = store.load_vectors(cfg.data_dir, video_id)
    if (
        not force
        and stored
        and stored.get("model") == cfg.embed_model
        and len(stored.get("vectors", [])) == len(chunks)
    ):
        return stored["vectors"]
    vectors = build_vectors(
        chunks, llm, embed_model=cfg.embed_model, progress=progress,
    )
    store.save_vectors(cfg.data_dir, video_id, cfg.embed_model, vectors, len(chunks))
    return vectors


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

    llm = client_for(cfg)
    _emit(progress, "indexing for search", 0.68)
    try:
        _ensure_vectors(
            cfg, llm, video.video_id, chunks,
            progress=lambda m, p: _emit(progress, m, 0.68 + 0.01 * p),
        )
    except LLMError as exc:
        _emit(progress, f"vector index skipped: {exc}", 0.69)

    summary = None if force else store.load_summary(cfg.data_dir, video.video_id)
    if summary is None:
        _emit(progress, "summarizing", 0.7)
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
    vectors = _ensure_vectors(
        cfg, llm, video_id, chunks, progress=progress, force=force_vectors,
    )

    _emit(progress, "asking", 0.3)
    answer: Answer = answer_question(
        video, chunks, question, llm,
        vectors=vectors, top_k=cfg.top_k, model=cfg.model,
        embed_model=cfg.embed_model, progress=progress,
    )
    record = store.video_record(data_dir, video_id) or {}
    record["data_dir"] = data_dir
    return {"answer": answer.to_dict(), "video": record}


GLOBAL_REFS = ("", "global", "__global__", "all", "__all__")


def _scope_for_ref(ref: str | None, cfg: Config) -> tuple[str, str]:
    cleaned = (ref or "").strip()
    if cleaned in GLOBAL_REFS:
        return "global", ""
    video_id = resolve_video_id(cleaned, cfg)
    if video_id is None:
        raise LLMError(f"no processed video found for '{cleaned}'")
    return "video", video_id


def _title_from(text: str) -> str:
    collapsed = " ".join((text or "").split())
    if not collapsed:
        return "New conversation"
    return collapsed[:60] + ("…" if len(collapsed) > 60 else "")


def create_thread(
    ref: str | None = None,
    *,
    title: str = "",
    mode: str = "ask",
    out_dir: str | None = None,
) -> dict[str, Any]:
    """Create a conversation thread for one video, or globally."""
    cfg = Config.load(out_dir)
    scope, video_id = _scope_for_ref(ref, cfg)
    thread = ChatThread(
        thread_id=uuid.uuid4().hex[:12],
        scope=scope,
        video_id=video_id,
        title=(title or "").strip(),
        mode="chat" if mode == "chat" else "ask",
        created=now(),
        updated=now(),
    )
    store.save_thread(cfg.data_dir, thread)
    return thread.to_dict()


def list_threads(
    ref: str | None = None, *, out_dir: str | None = None,
) -> list[dict[str, Any]]:
    cfg = Config.load(out_dir)
    scope, video_id = _scope_for_ref(ref, cfg)
    if scope == "video":
        store.migrate_answers(cfg.data_dir, video_id)
        return store.list_threads(cfg.data_dir, video_id=video_id)
    return store.list_threads(cfg.data_dir, scope="global")


def get_thread(thread_id: str, *, out_dir: str | None = None) -> dict[str, Any]:
    cfg = Config.load(out_dir)
    thread = store.load_thread(cfg.data_dir, thread_id)
    if thread is None:
        raise LLMError(f"no conversation '{thread_id}'")
    return thread.to_dict()


def delete_thread(thread_id: str, *, out_dir: str | None = None) -> bool:
    cfg = Config.load(out_dir)
    return store.delete_thread(cfg.data_dir, thread_id)


def update_thread(
    thread_id: str,
    *,
    title: str | None = None,
    playlist_ids: list[str] | None = None,
    mode: str | None = None,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """Edit a thread's title, mode, or playlist scope for global search."""
    cfg = Config.load(out_dir)
    thread = store.load_thread(cfg.data_dir, thread_id)
    if thread is None:
        raise LLMError(f"no conversation '{thread_id}'")
    if title is not None and title.strip():
        thread.title = title.strip()[:120]
    if playlist_ids is not None:
        known = {p.playlist_id for p in store.list_playlists(cfg.data_dir)}
        thread.playlist_ids = [pid for pid in playlist_ids if pid in known]
    if mode in ("ask", "chat"):
        thread.mode = mode
    thread.updated = now()
    store.save_thread(cfg.data_dir, thread)
    return thread.to_dict()


def _playlist_record(playlist: Playlist) -> dict[str, Any]:
    record = playlist.to_dict()
    record["video_count"] = len(playlist.video_ids)
    return record


def create_playlist(name: str, *, out_dir: str | None = None) -> dict[str, Any]:
    name = (name or "").strip()
    if not name:
        raise LLMError("playlist name is required")
    cfg = Config.load(out_dir)
    playlist = Playlist(
        playlist_id=uuid.uuid4().hex[:12],
        name=name,
        created=now(),
        updated=now(),
    )
    store.save_playlist(cfg.data_dir, playlist)
    return _playlist_record(playlist)


def list_playlists(*, out_dir: str | None = None) -> list[dict[str, Any]]:
    cfg = Config.load(out_dir)
    return [_playlist_record(p) for p in store.list_playlists(cfg.data_dir)]


def get_playlist(playlist_id: str, *, out_dir: str | None = None) -> dict[str, Any]:
    cfg = Config.load(out_dir)
    playlist = store.load_playlist(cfg.data_dir, playlist_id)
    if playlist is None:
        raise LLMError(f"no playlist '{playlist_id}'")
    return _playlist_record(playlist)


def update_playlist(
    playlist_id: str,
    *,
    name: str | None = None,
    add_video_ids: list[str] | None = None,
    remove_video_ids: list[str] | None = None,
    out_dir: str | None = None,
) -> dict[str, Any]:
    cfg = Config.load(out_dir)
    playlist = store.load_playlist(cfg.data_dir, playlist_id)
    if playlist is None:
        raise LLMError(f"no playlist '{playlist_id}'")
    if name is not None and name.strip():
        playlist.name = name.strip()
    removed = {str(v) for v in remove_video_ids or []}
    if removed:
        playlist.video_ids = [v for v in playlist.video_ids if v not in removed]
    for ref in add_video_ids or []:
        video_id = resolve_video_id(str(ref), cfg)
        if video_id and video_id not in playlist.video_ids:
            playlist.video_ids.append(video_id)
    playlist.updated = now()
    store.save_playlist(cfg.data_dir, playlist)
    return _playlist_record(playlist)


def delete_playlist(playlist_id: str, *, out_dir: str | None = None) -> bool:
    cfg = Config.load(out_dir)
    return store.delete_playlist(cfg.data_dir, playlist_id)


def _index_missing_vectors(
    cfg: Config,
    llm: OllamaClient,
    *,
    progress: ProgressFn | None = None,
    video_ids: list[str] | None = None,
) -> None:
    """Embed chunks for any video missing (or holding stale) vectors."""
    root = Path(cfg.data_dir)
    if not root.exists():
        return
    wanted = set(video_ids) if video_ids is not None else None
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not (child / store.METADATA).exists():
            continue
        if wanted is not None and child.name not in wanted:
            continue
        video_id = child.name
        chunks = store.load_chunks(cfg.data_dir, video_id) or []
        if not chunks:
            continue
        stored = store.load_vectors(cfg.data_dir, video_id)
        if (
            stored
            and stored.get("model") == cfg.embed_model
            and len(stored.get("vectors", [])) == len(chunks)
        ):
            continue
        _emit(progress, f"indexing {video_id} for search", 0.1)
        try:
            _ensure_vectors(
                cfg, llm, video_id, chunks,
                progress=lambda m, p: _emit(progress, m, 0.1 + 0.2 * p),
            )
        except LLMError:
            continue


def _thread_scope(
    cfg: Config, thread: ChatThread,
) -> tuple[list[str] | None, str | None]:
    """Resolve a global thread's playlist scope to video ids and a label."""
    if thread.scope != "global":
        return None, None
    if not thread.playlist_ids:
        return None, "the whole video library"
    playlists = [
        p for p in store.list_playlists(cfg.data_dir)
        if p.playlist_id in thread.playlist_ids
    ]
    if not playlists:
        return [], "no videos (deleted playlists)"
    videos = store.playlist_video_ids(cfg.data_dir, thread.playlist_ids)
    label = "playlists: " + ", ".join(p.name for p in playlists)
    return videos, label


def _retrieve_for_thread(
    thread: ChatThread,
    query: str,
    cfg: Config,
    llm: OllamaClient,
    progress: ProgressFn | None,
) -> list[Citation]:
    if thread.scope == "global":
        video_ids, _label = _thread_scope(cfg, thread)
        _emit(progress, "searching the library", 0.1)
        _index_missing_vectors(cfg, llm, progress=progress, video_ids=video_ids)
        return search_all(
            query,
            data_dir=cfg.data_dir,
            llm=llm,
            top_k=cfg.top_k,
            embed_model=cfg.embed_model,
            video_ids=video_ids,
        )
    chunks = store.load_chunks(cfg.data_dir, thread.video_id) or []
    if not chunks:
        transcript = store.load_transcript(cfg.data_dir, thread.video_id)
        if transcript is not None:
            chunks = chunks_from_transcript(transcript)
            store.save_chunks(cfg.data_dir, thread.video_id, chunks)
    if not chunks:
        raise LLMError(f"'{thread.video_id}' has no transcript; process it first")
    _emit(progress, "retrieving relevant moments", 0.1)
    vectors = _ensure_vectors(
        cfg, llm, thread.video_id, chunks,
        progress=lambda m, p: _emit(progress, m, 0.1 + 0.2 * p),
    )
    return search(
        query, chunks, vectors, llm, top_k=cfg.top_k, embed_model=cfg.embed_model,
    )


def post_message(
    thread_id: str,
    message: str,
    *,
    mode: str | None = None,
    out_dir: str | None = None,
    top_k: int | None = None,
    model: str | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Append a user turn, generate a reply with history, persist both."""
    message = (message or "").strip()
    if not message:
        raise LLMError("message is required")
    cfg = Config.load(out_dir)
    if model:
        cfg.model = model
        cfg.save()
    if top_k:
        cfg.top_k = top_k
    thread = store.load_thread(cfg.data_dir, thread_id)
    if thread is None:
        raise LLMError(f"no conversation '{thread_id}'")
    effective_mode = mode if mode in ("ask", "chat") else (thread.mode or "ask")

    llm = client_for(cfg)
    citations = _retrieve_for_thread(thread, message, cfg, llm, progress)
    video = (
        store.load_video(cfg.data_dir, thread.video_id)
        if thread.scope == "video"
        else None
    )
    _scope_ids, scope_label = _thread_scope(cfg, thread)

    thread.messages.append(
        ChatMessage(role="user", content=message, mode=effective_mode, created=now())
    )
    _emit(progress, "thinking", 0.5)
    body, usage = generate_reply_detailed(
        message,
        mode=effective_mode,
        history=thread.messages[:-1],
        citations=citations,
        llm=llm,
        video=video,
        global_scope=thread.scope == "global",
        scope_label=scope_label,
        model=cfg.model,
    )
    context_limit = (
        llm.loaded_context_length(cfg.model) or llm.model_context_length(cfg.model)
    )
    reply = ChatMessage(
        role="assistant",
        content=body or "The model returned an empty reply.",
        mode=effective_mode,
        citations=citations,
        model=cfg.model,
        created=now(),
        prompt_tokens=usage.get("prompt_tokens", 0),
        eval_tokens=usage.get("eval_tokens", 0),
        context_limit=context_limit,
    )
    thread.messages.append(reply)
    thread.mode = effective_mode
    thread.updated = now()
    if not thread.title.strip():
        thread.title = _title_from(message)
    store.save_thread(cfg.data_dir, thread)
    _emit(progress, "done", 1.0)
    return {
        "thread": thread.to_dict(),
        "message": reply.to_dict(),
        "context": context_status(thread.thread_id, out_dir=out_dir),
    }


def search_library(
    query: str,
    *,
    out_dir: str | None = None,
    top_k: int | None = None,
    playlist_ids: list[str] | None = None,
) -> dict[str, Any]:
    cfg = Config.load(out_dir)
    llm = client_for(cfg)
    video_ids = None
    if playlist_ids:
        video_ids = store.playlist_video_ids(cfg.data_dir, playlist_ids)
    _index_missing_vectors(cfg, llm, progress=None, video_ids=video_ids)
    citations = search_all(
        query,
        data_dir=cfg.data_dir,
        llm=llm,
        top_k=top_k or cfg.top_k,
        embed_model=cfg.embed_model,
        video_ids=video_ids,
    )
    return {
        "query": query,
        "citations": [c.to_dict() for c in citations],
        "playlist_ids": list(playlist_ids or []),
    }


def context_status(
    thread_id: str | None = None, *, out_dir: str | None = None,
) -> dict[str, Any]:
    """How full the context window is, from Ollama's own numbers."""
    cfg = Config.load(out_dir)
    llm = client_for(cfg)
    loaded = int(llm.loaded_context_length(cfg.model))
    maximum = int(llm.model_context_length(cfg.model))
    payload: dict[str, Any] = {
        "model": cfg.model,
        "server_ok": bool(loaded or maximum),
        "loaded_context": loaded,
        "max_context": maximum,
        "context_limit": loaded or maximum,
        "prompt_tokens": 0,
        "eval_tokens": 0,
        "thread_tokens": 0,
        "thread_messages": 0,
        "thread_id": thread_id or "",
    }
    if not thread_id:
        return payload
    thread = store.load_thread(cfg.data_dir, thread_id)
    if thread is None:
        return payload
    payload["thread_messages"] = len(thread.messages)
    payload["thread_tokens"] = sum(
        estimate_tokens(message.content)
        + sum(estimate_tokens(c.text) for c in message.citations)
        for message in thread.messages
    )
    for message in reversed(thread.messages):
        if message.role == "assistant" and message.prompt_tokens:
            payload["prompt_tokens"] = message.prompt_tokens
            payload["eval_tokens"] = message.eval_tokens
            if message.context_limit:
                payload["context_limit"] = message.context_limit
            break
    return payload


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
    store.delete_threads_for_video(cfg.data_dir, video_id)
    store.remove_video_from_playlists(cfg.data_dir, video_id)
    return store.delete_video(cfg.data_dir, video_id)


def answer_history(video_id: str, *, out_dir: str | None = None) -> list[dict[str, Any]]:
    cfg = Config.load(out_dir)
    return store.load_answers(cfg.data_dir, video_id)
