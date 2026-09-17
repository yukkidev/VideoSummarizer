"""Charters and agent factories."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from agents.framework import Agent, Charter, HumanGate
from agents.tools import build_tools
from pipeline.config import Config
from pipeline.llm import OllamaClient, client_from_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_PATHS = [
    "agents/framework.py",
    "pipeline/models.py",
    "pipeline/config.py",
    "pyproject.toml",
]

LOCAL_TOOLS = {"list_videos", "get_video", "ask_video", "read_file", "list_dir"}
PROJECT_WRITE_PATHS = [
    "data/**",
    "sandbox/**",
    "pipeline/**",
    "agents/**",
    "gui/**",
    "cli/**",
    "tests/**",
    "*.md",
]

ANALYST = Charter(
    role="Analyst",
    description=(
        "Answers questions about already-processed videos using local data. "
        "Has no network access and never modifies files."
    ),
    allowed_tools=set(LOCAL_TOOLS),
    allowed_write_paths=[],
    core_paths=[],
    max_steps=8,
)

RESEARCHER = Charter(
    role="Researcher",
    description=(
        "Finds and summarizes video sources. May search the web and run the "
        "video pipeline (gated). Does not edit project code."
    ),
    allowed_tools=LOCAL_TOOLS | {"search_web", "summarize_video"},
    allowed_write_paths=["data/**"],
    core_paths=CORE_PATHS,
    max_steps=12,
)

CODER = Charter(
    role="Coder",
    description=(
        "Writes and tests code. May edit project files, but edits to core "
        "files and model downloads require explicit approval."
    ),
    allowed_tools=LOCAL_TOOLS | {
        "search_web", "write_file", "edit_code", "run_command",
        "download_model", "summarize_video",
    },
    allowed_write_paths=PROJECT_WRITE_PATHS,
    core_paths=CORE_PATHS,
    max_steps=16,
)


@dataclass
class AgentOptions:
    out_dir: str = "data"
    model: str = ""
    gate: HumanGate = None  # type: ignore[assignment]
    llm: OllamaClient = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.llm is None:
            cfg = Config.load(self.out_dir)
            self.llm = client_from_config(cfg)
            if not self.model:
                self.model = cfg.model
        if self.gate is None:
            audit = Path(self.out_dir) / "agents" / "audit.jsonl"
            self.gate = HumanGate(audit_path=audit)


def _make(
    name: str, charter: Charter, options: AgentOptions,
    extra_tools: set[str] | None = None,
) -> Agent:
    available = build_tools(out_dir=options.out_dir)
    allowed = set(charter.allowed_tools)
    if extra_tools:
        allowed |= extra_tools
    tool_dict = {key: tool for key, tool in available.items() if key in allowed}
    return Agent(
        name=name,
        charter=replace(charter, allowed_tools=allowed),
        tools=tool_dict,
        gate=options.gate,
        llm=options.llm,
        model=options.model,
        max_steps=charter.max_steps,
    )


def make_analyst(options: AgentOptions) -> Agent:
    return _make("analyst", ANALYST, options)


def make_researcher(options: AgentOptions) -> Agent:
    return _make("researcher", RESEARCHER, options)


def make_coder(options: AgentOptions) -> Agent:
    return _make("coder", CODER, options)


AGENTS = {
    "analyst": (ANALYST, make_analyst),
    "researcher": (RESEARCHER, make_researcher),
    "coder": (CODER, make_coder),
}


def build_agent(name: str, options: AgentOptions) -> Agent:
    try:
        factory = AGENTS[name][1]
    except KeyError as exc:
        raise KeyError(f"unknown agent '{name}'; choose from {', '.join(AGENTS)}") from exc
    return factory(options)
