"""Agent layer entry point: safety demo and live LLM tool loop.

Examples
--------
python -m agents.demo                                   # safety demo (no LLM)
python -m agents.demo --agent analyst --task "..."      # real tool-use loop
python -m agents.demo --url "https://..." --interactive # summarize via gated tool
"""

from __future__ import annotations

import argparse
import json

from agents.framework import (
    CharterViolation,
    GateDenied,
    HumanGate,
    truncate,
)
from agents.registry import AgentOptions, build_agent


def safety_demo(options: AgentOptions) -> int:
    print("HumanGate initialized. Guarded/dangerous actions are denied unless")
    print("--interactive or --auto is passed.\n")

    analyst = build_agent("analyst", options)
    coder = build_agent("coder", options)

    print("1) ANALYST reads its own charter tool list (safe, auto-runs):")
    print("   tools:", ", ".join(sorted(analyst.tools)))

    print("\n2) ANALYST tries summarize_video (not in charter):")
    try:
        analyst.action("summarize_video", url="https://example.com/v")
        print("   -> unexpectedly allowed")
    except CharterViolation as exc:
        print(f"   -> CHARTER BLOCKED: {exc}")

    print("\n3) CODER tries a network search (guarded, gate denies by default):")
    try:
        coder.action("search_web", query="test")
        print("   -> allowed")
    except GateDenied as exc:
        print(f"   -> GATE DENIED: {exc}")

    print("\n4) CODER tries to edit a core file (dangerous):")
    try:
        coder.action("edit_code", path="agents/framework.py", old="x", new="y")
        print("   -> allowed")
    except GateDenied as exc:
        print(f"   -> GATE DENIED: {exc}")

    print("\n5) CODER writes inside data/ (guarded, still gated non-interactively):")
    try:
        coder.action("write_file", path="data/demo_note.txt", content="hello")
        print("   -> allowed")
    except GateDenied as exc:
        print(f"   -> GATE DENIED: {exc}")

    print("\nGate decisions:")
    for decision in coder.decision_log():
        verdict = "APPROVED" if decision["approved"] else "DENIED"
        print(f"   - {decision['action']} by {decision['by']}: {verdict}")
    if options.gate.audit_path:
        print(f"\naudit log: {options.gate.audit_path}")
    return 0


def print_trace(agent, final: str) -> None:
    print("\ntrace:")
    for entry in agent.trace:
        if entry["kind"] == "action":
            verdict = "approved" if entry["approved"] else "denied"
            args = json.dumps(entry.get("args", {}), default=str)
            print(f"   - {entry['tool']} ({entry['danger']}, {verdict}) {truncate(args, 120)}")
        elif entry["kind"] == "final":
            pass
        elif entry["kind"] == "timeout":
            print(f"   - hit the {entry['steps']}-step limit")
    print("\nfinal answer:\n" + final)


def run_agent(args: argparse.Namespace) -> int:
    gate = HumanGate(
        auto_approve=args.auto,
        interactive=args.interactive,
        audit_path=args.out_dir + "/agents/audit.jsonl",
    )
    options = AgentOptions(out_dir=args.out_dir, gate=gate)
    if args.model:
        options.model = args.model
    agent = build_agent(args.agent, options)
    if args.model:
        agent.model = args.model

    if args.url:
        task = (
            "Summarize this video and list its key questions: " + args.url
        )
    else:
        task = args.task or (
            "List the processed videos and summarize the most recent one's "
            "main points in a few sentences."
        )

    print(f"[{agent.name}] task: {task}")
    print(f"[{agent.name}] tools: {', '.join(sorted(agent.tools))}\n")
    try:
        final = agent.run(task, max_steps=args.max_steps or None)
    except Exception as exc:  # noqa: BLE001 - surface any agent failure to the user
        print(f"agent failed: {exc}")
        return 1
    if args.show_trace or args.auto or args.interactive:
        print_trace(agent, final)
    else:
        print(final)
    decisions = agent.decision_log()
    if decisions:
        print("\ngate decisions:")
        for decision in decisions:
            verdict = "APPROVED" if decision["approved"] else "DENIED"
            print(f"   - {decision['action']}: {verdict}")
    if gate.audit_path:
        print(f"\naudit log: {gate.audit_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vidsum-agents",
        description="VideoSummarizer agent layer",
    )
    parser.add_argument("--agent", default="analyst", choices=["analyst", "researcher", "coder"])
    parser.add_argument("--task", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--auto", action="store_true",
                        help="auto-approve gated actions (automation only)")
    parser.add_argument("--interactive", action="store_true",
                        help="prompt for each gated action")
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--demo", action="store_true",
                        help="run the scripted safety demo (default when no task/url)")
    args = parser.parse_args(argv)

    gate = HumanGate(
        auto_approve=args.auto,
        interactive=args.interactive,
        audit_path=args.out_dir + "/agents/audit.jsonl",
    )
    options = AgentOptions(out_dir=args.out_dir, gate=gate, model=args.model)

    if args.demo or (not args.task and not args.url):
        return safety_demo(options)
    return run_agent(args)


if __name__ == "__main__":
    raise SystemExit(main())
