"""LLM-driven multi-role agent framework with charters and a human gate.

Design
------
- Every tool has a danger level: ``safe``, ``guarded``, or ``dangerous``.
- Safe tools run freely. Everything else goes through :class:`HumanGate`.
- An :class:`Agent` has a :class:`Charter` the framework enforces: allowed
  tools, allowed write paths, and protected core paths.
- ``Agent.run()`` is a real tool-use loop backed by a local LLM (Ollama).
- Every gate decision is appended to an audit log (JSONL).
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.llm import OllamaClient, strip_thinking

SAFE = "safe"
GUARDED = "guarded"
DANGEROUS = "dangerous"
DANGER_ORDER = {SAFE: 0, GUARDED: 1, DANGEROUS: 2}


class AgentError(RuntimeError):
    pass


class CharterViolation(AgentError):
    pass


class GateDenied(AgentError):
    pass


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    danger: str = SAFE
    path_arg: str | None = None
    params: dict[str, str] = field(default_factory=dict)

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "danger": self.danger,
            "parameters": self.params,
        }


@dataclass
class Charter:
    role: str
    description: str
    allowed_tools: set[str] = field(default_factory=set)
    allowed_write_paths: list[str] = field(default_factory=lambda: ["data/**", "sandbox/**"])
    core_paths: list[str] = field(default_factory=list)
    max_steps: int = 12

    def tool_allowed(self, name: str) -> bool:
        if "*" in self.allowed_tools:
            return True
        return name in self.allowed_tools

    def path_allowed(self, path: str) -> bool:
        resolved = str(Path(path).expanduser().resolve())
        for pattern in self.allowed_write_paths:
            if fnmatch.fnmatch(resolved, str(Path(pattern).expanduser().resolve())) or fnmatch.fnmatch(
                resolved, pattern,
            ):
                return True
        return False

    def is_core(self, path: str) -> bool:
        resolved = str(Path(path).expanduser().resolve())
        return any(
            fnmatch.fnmatch(resolved, pattern) or fnmatch.fnmatch(resolved, str(Path(pattern).expanduser().resolve()))
            for pattern in self.core_paths
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "description": self.description,
            "allowed_tools": sorted(self.allowed_tools),
            "allowed_write_paths": self.allowed_write_paths,
            "core_paths": self.core_paths,
            "max_steps": self.max_steps,
        }


@dataclass
class Approval:
    action: str
    by: str
    approved: bool
    note: str = ""
    auto: bool = False
    exit: bool = False
    at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "by": self.by,
            "approved": self.approved,
            "note": self.note,
            "auto": self.auto,
            "exit": self.exit,
            "at": self.at,
        }


class HumanGate:
    """Single choke point for guarded and dangerous actions.

    - safe actions always pass
    - non-interactive: destructive actions are denied (safe default)
    - interactive: prompt the user for a decision
    - auto_approve: explicit hands-off mode for automation
    """

    def __init__(
        self,
        *,
        auto_approve: bool = False,
        interactive: bool = False,
        audit_path: str | Path | None = None,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self.auto_approve = auto_approve
        self.interactive = interactive
        self.approvals: list[Approval] = []
        self.audit_path = Path(audit_path) if audit_path else None
        self._input = input_fn

    def request(self, action: str, *, by: str, danger: str = GUARDED, note: str = "") -> bool:
        if danger == SAFE:
            return True
        if self.auto_approve:
            self._record(action, by, True, note, auto=True)
            return True
        if self.interactive:
            return self.wait(action, by=by, note=note)
        self._record(action, by, False, note)
        return False

    def wait(self, action: str, *, by: str, note: str = "") -> bool:
        prompt = (
            f"\n[{by}] wants to run '{action}'"
            + (f" ({note})" if note else "")
            + " [a]pprove [d]eny [x]it: "
        )
        try:
            reply = self._input(prompt)
        except (EOFError, KeyboardInterrupt):
            reply = "d"
        answer = reply.strip().lower()
        if answer in ("a", "approve", "y", "yes"):
            self._record(action, by, True, note)
            return True
        exit_now = answer in ("x", "exit", "quit")
        self._record(action, by, False, note, exit=exit_now)
        return False

    def _record(
        self, action: str, by: str, approved: bool, note: str,
        *, auto: bool = False, exit: bool = False,
    ) -> None:
        approval = Approval(
            action=action,
            by=by,
            approved=approved,
            note=note,
            auto=auto,
            exit=exit,
            at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.approvals.append(approval)
        if self.audit_path:
            try:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(approval.to_dict(), ensure_ascii=False) + "\n")
            except OSError:
                pass

    @property
    def exited(self) -> bool:
        return any(a.exit for a in self.approvals)


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
TOOL_KEYS = ("tool", "action", "name", "tool_name")
ARG_KEYS = ("args", "arguments", "parameters", "input")
FINAL_KEYS = ("final", "final_answer", "answer", "response")


@dataclass
class ParsedCall:
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    final: str | None = None


def parse_tool_call(text: str) -> ParsedCall:
    """Tolerantly extract a tool call or final answer from model output."""
    cleaned = strip_thinking(text or "")
    match = JSON_OBJECT_RE.search(cleaned)
    data: Any = None
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
    if data is None:
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return ParsedCall()
    if not isinstance(data, dict):
        return ParsedCall(final=cleaned)
    if len(data) == 1:
        key, value = next(iter(data.items()))
        if isinstance(value, str) and key.lower() in FINAL_KEYS:
            return ParsedCall(final=value)

    tool = None
    for key in TOOL_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            tool = value.strip()
            break
    if tool is None:
        for key in FINAL_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return ParsedCall(final=value.strip())
        return ParsedCall()

    args: dict[str, Any] = {}
    for key in ARG_KEYS:
        value = data.get(key)
        if isinstance(value, dict):
            args = value
            break
        if isinstance(value, str) and value.strip():
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                args = decoded
                break
    if not args:
        args = {
            key: value for key, value in data.items()
            if key not in TOOL_KEYS and key not in FINAL_KEYS and key not in ARG_KEYS
        }
    return ParsedCall(tool=tool, args=args)


def truncate(text: str, limit: int = 1500) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


@dataclass
class Agent:
    name: str
    charter: Charter
    tools: dict[str, Tool] = field(default_factory=dict)
    gate: HumanGate = field(default_factory=HumanGate)
    llm: OllamaClient | None = None
    model: str = ""
    max_steps: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def role(self) -> str:
        return self.charter.role

    @property
    def description(self) -> str:
        return self.charter.description

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def tool_specs(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def _check_charter(self, name: str, args: dict[str, Any]) -> Tool:
        if not self.charter.tool_allowed(name):
            raise CharterViolation(
                f"tool '{name}' is not in the charter for '{self.name}'"
            )
        tool = self.tools.get(name)
        if tool is None:
            raise CharterViolation(f"agent '{self.name}' has no tool '{name}'")
        if tool.path_arg and tool.path_arg in args:
            path = str(args[tool.path_arg])
            if tool.danger in (GUARDED, DANGEROUS) and not self.charter.path_allowed(path):
                raise CharterViolation(
                    f"path '{path}' is outside the charter's write paths"
                )
            if self.charter.is_core(path) and tool.danger != SAFE:
                tool = Tool(
                    name=tool.name + "@core",
                    description=tool.description + " (core file)",
                    func=tool.func,
                    danger=DANGEROUS,
                    path_arg=tool.path_arg,
                    params=tool.params,
                )
        return tool

    def action(self, name: str, **kwargs: Any) -> str:
        tool = self._check_charter(name, kwargs)
        approved = self.gate.request(
            tool.name, by=self.name, danger=tool.danger,
            note=tool.description[:80],
        )
        self.trace.append({
            "kind": "action",
            "tool": name,
            "args": kwargs,
            "approved": approved,
            "danger": tool.danger,
        })
        if not approved:
            raise GateDenied(
                f"action '{name}' was not approved by the human gate"
            )
        result = tool.func(**kwargs)
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, default=str)
        return str(result)

    def _system_prompt(self) -> str:
        return (
            f"You are {self.role}: {self.description}\n"
            "You work by calling tools and observing results, then producing a "
            "final answer. Respond with ONE JSON object per step.\n"
            'To call a tool: {"tool": "<name>", "args": {<arguments>}}\n'
            'To finish: {"final": "<your answer>"}\n'
            "Never invent tool results. Only use tools listed below.\n"
            "Never call the same tool twice with the same arguments. As soon as "
            "you have enough information, respond with {\"final\": ...} — do not "
            "keep exploring.\n"
            "Tools:\n"
            + json.dumps(self.tool_specs(), indent=2)
        )

    def _prompt(self, history: Sequence[str]) -> str:
        return (
            "Conversation so far:\n\n"
            + "\n\n".join(history)
            + "\n\nYour next JSON response:"
        )

    def run(self, task: str, max_steps: int | None = None) -> str:
        if self.llm is None:
            raise AgentError("no LLM configured for this agent")
        steps = max_steps or self.max_steps or self.charter.max_steps
        history: list[str] = [f"TASK: {task}"]
        seen: dict[str, str] = {}
        for _ in range(steps):
            raw = self.llm.generate(
                self._prompt(history),
                model=self.model or self.llm.model,
                system=self._system_prompt(),
                json_mode=True,
                temperature=0.1,
            )
            parsed = parse_tool_call(raw)
            if parsed.final is not None:
                self.trace.append({"kind": "final", "text": parsed.final})
                return parsed.final
            if not parsed.tool:
                history.append(
                    "OBSERVATION: your reply was not a valid JSON tool call. "
                    "Respond with {\"tool\": ..., \"args\": {...}} or {\"final\": ...}."
                )
                continue
            signature = f"{parsed.tool}|{json.dumps(parsed.args, sort_keys=True, default=str)}"
            if signature in seen:
                history.append(
                    f"OBSERVATION: you already called {signature} and received:\n"
                    f"{seen[signature]}\n"
                    "You have the information you need. Respond now with "
                    "{\"final\": \"<answer>\"}."
                )
                continue
            try:
                result = self.action(parsed.tool, **parsed.args)
                observation = truncate(result)
            except AgentError as exc:
                observation = f"DENIED: {exc}"
            except Exception as exc:  # noqa: BLE001 - tool errors become observations
                observation = f"TOOL ERROR: {type(exc).__name__}: {exc}"
            seen[signature] = observation
            history.append(
                f"ACTION: {parsed.tool} {json.dumps(parsed.args, default=str)}\n"
                f"OBSERVATION: {observation}"
            )
        self.trace.append({"kind": "timeout", "steps": steps})
        return (
            f"Agent '{self.name}' reached the {steps}-step limit without a "
            "final answer."
        )

    def decision_log(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.gate.approvals]
