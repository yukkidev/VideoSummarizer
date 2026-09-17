"""Back-compat entry point: ``python -m pipeline.runner <url>``.

Prefer ``vidsum run <url>``; this module delegates to the same facade.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from pipeline import api


def run(
    url: str,
    *,
    download_audio: bool = True,
    out_dir: str = "data",
    model: str | None = None,
    verbose: bool = True,
    progress=None,
) -> dict[str, Any]:
    record = api.process(
        url,
        out_dir=out_dir,
        want_audio=download_audio,
        model=model,
        progress=progress,
    )
    if verbose:
        summary = record.get("summary") or {}
        print("=" * 60)
        print(f"VIDEO: {record.get('title')}")
        print(f"AUTHOR: {record.get('author')}")
        print("=" * 60)
        print("\nSUMMARY:\n" + (summary.get("summary") or ""))
        print("\nTOP QUESTIONS:")
        for q in summary.get("highlights", []):
            print(f"- {q}")
        print("=" * 60)
        print(f"artifacts: {record.get('dir')}")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local video summarizer + Q&A.",
    )
    parser.add_argument("url", help="video URL to process")
    parser.add_argument("--no-audio", action="store_true", help="skip audio download")
    parser.add_argument("--no-video", action="store_true", help="skip video download")
    parser.add_argument("--subs-only", action="store_true", help="subtitles only, no downloads")
    parser.add_argument("--force", action="store_true", help="ignore caches")
    parser.add_argument("--model", default=None, help="Ollama model name")
    parser.add_argument("--whisper-model", default=None, help="faster-whisper model")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--json", action="store_true", help="print result JSON to stdout")
    args = parser.parse_args(argv)

    subs_only = args.subs_only or args.no_audio
    try:
        record = api.process(
            args.url,
            out_dir=args.out_dir,
            want_audio=not subs_only,
            want_video=not (args.no_video or subs_only),
            want_subtitles=True,
            force=args.force,
            model=args.model,
            whisper_model=args.whisper_model,
            force_subtitles=args.subs_only,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        json.dump(record, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        summary = record.get("summary") or {}
        print("=" * 60)
        print(f"VIDEO: {record.get('title')}")
        print(f"AUTHOR: {record.get('author')}")
        print(f"SOURCE: {record.get('transcript_source')}")
        print("=" * 60)
        print("\nSUMMARY:\n" + (summary.get("summary") or ""))
        print("\nTOP QUESTIONS:")
        for q in summary.get("highlights", []):
            print(f"- {q}")
        print("=" * 60)
        print(f"artifacts: {record.get('dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
