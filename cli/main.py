"""vidsum command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from pipeline import api
from pipeline.llm import LLMError
from pipeline.models import format_timestamp


def _print_json(payload: Any) -> None:
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _progress_printer(quiet: bool):
    last = {"msg": ""}

    def report(message: str, fraction: float) -> None:
        if quiet:
            return
        if message != last["msg"]:
            last["msg"] = message
            print(f"  [{int(fraction * 100):>3}%] {message}", file=sys.stderr)

    return report


def cmd_run(args: argparse.Namespace) -> int:
    try:
        record = api.process(
            args.url,
            out_dir=args.out_dir,
            want_audio=not args.no_audio,
            want_video=not args.no_video,
            force=args.force,
            model=args.model,
            whisper_model=args.whisper_model,
            force_subtitles=args.subtitles,
            progress=_progress_printer(args.quiet),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print_json(record)
        return 0
    summary = record.get("summary") or {}
    print(f"\n{record.get('title')} — {record.get('author')}")
    print(f"transcript: {record.get('transcript_source')} | "
          f"segments: {record.get('segment_count')} | model: {summary.get('model')}")
    print("\nSUMMARY\n" + (summary.get("summary") or "(none)"))
    print("\nQUESTIONS TO EXPLORE")
    for q in summary.get("highlights", []):
        print(f"  - {q}")
    print(f"\nartifacts: {record.get('dir')}")
    return 0


def _print_answer(payload: dict[str, Any]) -> None:
    answer = payload.get("answer") or {}
    video = payload.get("video") or {}
    print(f"\nQ: {answer.get('question')}")
    print(f"\nA: {answer.get('answer')}\n")
    citations = answer.get("citations") or []
    if citations:
        print("Sources:")
        for citation in citations:
            stamp = f"{format_timestamp(citation.get('start', 0))}"
            print(f"  [{stamp}] score={citation.get('score', 0):.2f} "
                  f"{citation.get('text', '')[:140]}")
    if video.get("video_path"):
        print(f"\nvideo file: {video['video_path']}")


def cmd_ask(args: argparse.Namespace) -> int:
    question = " ".join(args.question).strip()
    if question:
        try:
            payload = api.ask(
                args.ref, question,
                out_dir=args.out_dir, top_k=args.top_k, model=args.model,
                progress=_progress_printer(args.quiet),
            )
        except LLMError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            _print_json(payload)
        else:
            _print_answer(payload)
        return 0

    print("Interactive Q&A. Type /quit to exit, /sources for last sources.\n")
    last: dict[str, Any] | None = None
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            return 0
        if line == "/sources" and last:
            _print_answer(last)
            continue
        try:
            last = api.ask(
                args.ref, line,
                out_dir=args.out_dir, top_k=args.top_k, model=args.model,
            )
        except LLMError as exc:
            print(f"error: {exc}", file=sys.stderr)
            continue
        answer = last.get("answer") or {}
        print(f"\n{answer.get('answer')}\n")
        for citation in answer.get("citations") or []:
            stamp = format_timestamp(citation.get("start", 0))
            print(f"  [{stamp}] {citation.get('text', '')[:120]}")
        print()


def cmd_list(args: argparse.Namespace) -> int:
    videos = api.list_videos(out_dir=args.out_dir)
    if args.json:
        _print_json(videos)
        return 0
    if not videos:
        print("no videos processed yet")
        return 0
    for video in videos:
        flags = []
        for key, label in (
            ("has_video", "video"),
            ("has_audio", "audio"),
            ("has_subtitles", "subs"),
            ("has_transcript", "transcript"),
            ("has_summary", "summary"),
            ("has_vectors", "vectors"),
        ):
            if video.get(key):
                flags.append(label)
        print(f"{video.get('video_id'):<14} {video.get('title', '')[:50]:<52} "
              f"[{', '.join(flags)}]")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    try:
        if args.set:
            status = api.set_model(args.set, data_dir=args.out_dir)
        else:
            status = api.models_status(args.model, config=api.load_config(args.out_dir))
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print_json(status)
        return 0
    print(f"server: {'ok' if status.get('server_ok') else 'DOWN'}")
    print(f"current: {status.get('current')} "
          f"({'loaded in memory' if status.get('current_loaded') else 'not loaded'})")
    if status.get("error"):
        print(f"note: {status['error']}")
    print("\ninstalled models:")
    for model in status.get("installed", []):
        marker = "*" if model["name"] == status.get("current") else " "
        print(f" {marker} {model['name']:<30} {model.get('size_gb', 0):>6} GB")
    loaded = status.get("loaded", [])
    if loaded:
        print("\nloaded in memory:")
        for model in loaded:
            print(f"   {model['name']:<30} {model.get('size_vram_gb', 0):>6} GB VRAM")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    if api.delete_video(args.ref, out_dir=args.out_dir):
        print(f"deleted {args.ref}")
        return 0
    print(f"not found: {args.ref}", file=sys.stderr)
    return 1


def cmd_tui(args: argparse.Namespace) -> int:
    from gui.tui import run_tui

    run_tui(out_dir=args.out_dir)
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from gui.server import run_server

    run_server(out_dir=args.out_dir, host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    from agents.demo import main as agents_main

    return agents_main(args.agent_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidsum",
        description="Local video downloader, transcriber, summarizer, and Q&A.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", default="data", help="data directory")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", parents=[common], help="download, transcribe, and summarize a video")
    run_p.add_argument("url")
    run_p.add_argument("--no-audio", action="store_true")
    run_p.add_argument("--no-video", action="store_true")
    run_p.add_argument("--subtitles", action="store_true",
                       help="use subtitles instead of audio transcription")
    run_p.add_argument("--force", action="store_true")
    run_p.add_argument("--model")
    run_p.add_argument("--whisper-model")
    run_p.add_argument("--json", action="store_true")
    run_p.add_argument("--quiet", action="store_true")
    run_p.set_defaults(func=cmd_run)

    ask_p = sub.add_parser("ask", parents=[common], help="ask questions about a processed video")
    ask_p.add_argument("ref", help="video id, data dir, or URL")
    ask_p.add_argument("question", nargs="*")
    ask_p.add_argument("--top-k", type=int)
    ask_p.add_argument("--model")
    ask_p.add_argument("--json", action="store_true")
    ask_p.add_argument("--quiet", action="store_true")
    ask_p.set_defaults(func=cmd_ask)

    list_p = sub.add_parser("list", parents=[common], help="list processed videos")
    list_p.add_argument("--json", action="store_true")
    list_p.set_defaults(func=cmd_list)

    models_p = sub.add_parser("models", parents=[common], help="list/refresh/switch Ollama models")
    models_p.add_argument("--refresh", action="store_true")
    models_p.add_argument("--set", metavar="MODEL")
    models_p.add_argument("--model", help="report status for this model")
    models_p.add_argument("--json", action="store_true")
    models_p.set_defaults(func=cmd_models)

    delete_p = sub.add_parser("delete", parents=[common], help="delete a processed video")
    delete_p.add_argument("ref")
    delete_p.set_defaults(func=cmd_delete)

    tui_p = sub.add_parser("tui", parents=[common], help="launch the terminal interface")
    tui_p.set_defaults(func=cmd_tui)

    gui_p = sub.add_parser("gui", parents=[common], help="launch the web GUI")
    gui_p.add_argument("--host", default="127.0.0.1")
    gui_p.add_argument("--port", type=int, default=8765)
    gui_p.add_argument("--no-browser", action="store_true")
    gui_p.set_defaults(func=cmd_gui)

    agents_p = sub.add_parser("agents", parents=[common], help="run the agent layer")
    agents_p.add_argument("agent_args", nargs=argparse.REMAINDER)
    agents_p.set_defaults(func=cmd_agents)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "out_dir"):
        pass
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
