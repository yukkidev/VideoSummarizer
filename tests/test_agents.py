from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.framework import (
    DANGEROUS,
    GUARDED,
    SAFE,
    Agent,
    AgentError,
    Charter,
    CharterViolation,
    GateDenied,
    HumanGate,
    Tool,
    parse_tool_call,
    truncate,
)
from agents.registry import AgentOptions, build_agent, make_analyst, make_coder
from agents.tools import _read_file, _run_command, build_tools

# ---- parsing ---------------------------------------------------------------

def test_parse_tool_call_basic():
    parsed = parse_tool_call('{"tool": "read_file", "args": {"path": "x"}}')
    assert parsed.tool == "read_file"
    assert parsed.args == {"path": "x"}
    assert parsed.final is None


def test_parse_tool_call_action_alias_and_parameters():
    parsed = parse_tool_call('{"action": "list_dir", "parameters": {"path": "."}}')
    assert parsed.tool == "list_dir"
    assert parsed.args == {"path": "."}


def test_parse_tool_call_args_from_flat_fields():
    parsed = parse_tool_call('{"tool": "read_file", "path": "a.txt"}')
    assert parsed.tool == "read_file"
    assert parsed.args == {"path": "a.txt"}


def test_parse_tool_call_ignores_non_dict_args():
    parsed = parse_tool_call('{"tool": "list_videos", "args": "none"}')
    assert parsed.tool == "list_videos"
    assert parsed.args == {}


def test_parse_tool_call_decodes_json_string_args():
    parsed = parse_tool_call('{"tool": "read_file", "args": "{\\"path\\": \\"a\\"}"}')
    assert parsed.args == {"path": "a"}


def test_parse_tool_call_final():
    assert parse_tool_call('{"final": "done"}').final == "done"
    assert parse_tool_call('{"answer": "done"}').final == "done"


def test_parse_tool_call_with_thinking_and_text():
    raw = "<" + "think" + ">hmm<" + "/" + "think" + ">\nHere: {\"tool\": \"x\", \"args\": {}}"
    assert parse_tool_call(raw).tool == "x"


def test_parse_tool_call_invalid():
    parsed = parse_tool_call("I have no idea")
    assert parsed.tool is None and parsed.final is None


def test_parse_tool_call_plain_string_json():
    assert parse_tool_call('"hello"').final == '"hello"'


def test_truncate():
    assert truncate("abc", 10) == "abc"
    assert truncate("abcdef", 3) == "abc …[truncated]"


# ---- human gate ------------------------------------------------------------

def make_gate(tmp_path: Path, **kwargs) -> HumanGate:
    kwargs.setdefault("audit_path", tmp_path / "audit.jsonl")
    return HumanGate(**kwargs)


def test_gate_safe_always_allows(tmp_path):
    gate = make_gate(tmp_path)
    assert gate.request("read_file", by="t", danger=SAFE) is True
    assert gate.approvals == []


def test_gate_denies_by_default(tmp_path):
    gate = make_gate(tmp_path)
    assert gate.request("download_model", by="t", danger=DANGEROUS) is False
    assert gate.approvals[-1].approved is False
    assert gate.approvals[-1].auto is False


def test_gate_auto_approve(tmp_path):
    gate = make_gate(tmp_path, auto_approve=True)
    assert gate.request("install_package", by="t", danger=DANGEROUS) is True
    assert gate.approvals[-1].auto is True
    assert gate.approvals[-1].approved is True


def test_gate_interactive_approve(tmp_path):
    answers = iter(["a"])
    gate = make_gate(tmp_path, interactive=True, input_fn=lambda _: next(answers))
    assert gate.request("write_file", by="t", danger=GUARDED) is True
    assert gate.approvals[-1].approved


def test_gate_interactive_deny_and_exit(tmp_path):
    answers = iter(["d", "x"])
    gate = make_gate(tmp_path, interactive=True, input_fn=lambda _: next(answers))
    assert gate.request("write_file", by="t") is False
    assert gate.request("run_command", by="t") is False
    assert gate.exited is True


def test_gate_interactive_eof_denies(tmp_path):
    def boom(prompt):
        raise EOFError

    gate = make_gate(tmp_path, interactive=True, input_fn=boom)
    assert gate.request("write_file", by="t") is False


def test_gate_audit_log_written(tmp_path):
    gate = make_gate(tmp_path, auto_approve=True)
    gate.request("download_model", by="coder", danger=DANGEROUS, note="notes")
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["action"] == "download_model"
    assert record["by"] == "coder"
    assert record["approved"] is True


# ---- charter ---------------------------------------------------------------

def test_charter_tool_and_path_rules():
    charter = Charter(
        role="r", description="d",
        allowed_tools={"read_file"},
        allowed_write_paths=["data/**"],
        core_paths=["agents/framework.py"],
    )
    assert charter.tool_allowed("read_file")
    assert not charter.tool_allowed("write_file")
    assert charter.path_allowed("data/x/y.txt")
    assert not charter.path_allowed("/etc/passwd")
    assert charter.is_core("agents/framework.py")
    assert not charter.is_core("agents/tools.py")


# ---- agent enforcement -----------------------------------------------------

def make_agent(tmp_path: Path, charter: Charter, **gate_kwargs) -> Agent:
    tools = {
        "read_file": Tool("read_file", "read", lambda path: f"content of {path}", SAFE, path_arg="path"),
        "write_file": Tool("write_file", "write", lambda path, content: f"wrote {path}", GUARDED, path_arg="path"),
        "edit_code": Tool("edit_code", "edit", lambda path, old, new: "edited", GUARDED, path_arg="path"),
        "danger": Tool("danger", "danger", lambda: "boom", DANGEROUS),
    }
    return Agent(
        name="tester",
        charter=charter,
        tools=tools,
        gate=HumanGate(**gate_kwargs),
        llm=None,
    )


def test_agent_safe_action_runs(tmp_path):
    charter = Charter(role="r", description="d", allowed_tools={"read_file"},
                      allowed_write_paths=["data/**"])
    agent = make_agent(tmp_path, charter)
    assert agent.action("read_file", path="data/x") == "content of data/x"


def test_agent_charter_blocks_unlisted_tool(tmp_path):
    charter = Charter(role="r", description="d", allowed_tools={"read_file"},
                      allowed_write_paths=["data/**"])
    agent = make_agent(tmp_path, charter)
    with pytest.raises(CharterViolation, match="not in the charter"):
        agent.action("write_file", path="data/x", content="c")


def test_agent_path_restriction(tmp_path):
    charter = Charter(role="r", description="d", allowed_tools={"write_file"},
                      allowed_write_paths=["data/**"])
    agent = make_agent(tmp_path, charter, auto_approve=True)
    with pytest.raises(CharterViolation, match="outside the charter"):
        agent.action("write_file", path="/etc/passwd", content="c")


def test_agent_gate_denies_guarded(tmp_path):
    charter = Charter(role="r", description="d", allowed_tools={"write_file"},
                      allowed_write_paths=["data/**"])
    agent = make_agent(tmp_path, charter)
    with pytest.raises(GateDenied):
        agent.action("write_file", path="data/x", content="c")


def test_agent_gate_allows_with_auto(tmp_path):
    charter = Charter(role="r", description="d", allowed_tools={"write_file"},
                      allowed_write_paths=["data/**"])
    agent = make_agent(tmp_path, charter, auto_approve=True)
    assert agent.action("write_file", path="data/x", content="c") == "wrote data/x"


def test_agent_core_path_upgrades_danger(tmp_path):
    charter = Charter(role="r", description="d", allowed_tools={"edit_code"},
                      allowed_write_paths=["agents/**"],
                      core_paths=["agents/framework.py"])
    agent = make_agent(tmp_path, charter, auto_approve=True)
    agent.action("edit_code", path="agents/framework.py", old="a", new="b")
    assert agent.trace[-1]["danger"] == DANGEROUS


def test_agent_trace_records_denials(tmp_path):
    charter = Charter(role="r", description="d", allowed_tools={"danger"})
    agent = make_agent(tmp_path, charter)
    with pytest.raises(GateDenied):
        agent.action("danger")
    assert agent.trace[-1]["approved"] is False
    assert len(agent.decision_log()) == 1


# ---- LLM tool loop ---------------------------------------------------------

class FakeLLM:
    model = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("no scripted responses left")
        return self.responses.pop(0)


def loop_agent(tmp_path, responses) -> Agent:
    charter = Charter(role="r", description="d",
                      allowed_tools={"read_file"}, allowed_write_paths=[])
    agent = make_agent(tmp_path, charter, auto_approve=True)
    agent.llm = FakeLLM(responses)
    return agent


def test_agent_run_calls_tool_then_finalizes(tmp_path):
    agent = loop_agent(tmp_path, [
        '{"tool": "read_file", "args": {"path": "data/a.txt"}}',
        '{"final": "the answer"}',
    ])
    assert agent.run("do it") == "the answer"
    kinds = [entry["kind"] for entry in agent.trace]
    assert kinds == ["action", "final"]
    assert agent.trace[0]["args"]["path"] == "data/a.txt"
    assert "content of data/a.txt" in agent.llm.prompts[1]


def test_agent_run_handles_malformed_then_final(tmp_path):
    agent = loop_agent(tmp_path, ["nonsense", '{"final": "recovered"}'])
    assert agent.run("t") == "recovered"
    assert agent.llm.prompts[1].find("not a valid JSON tool call") != -1


def test_agent_run_continues_after_denial(tmp_path):
    charter = Charter(role="r", description="d", allowed_tools={"danger"})
    agent = make_agent(tmp_path, charter)
    agent.llm = FakeLLM(['{"tool": "danger"}', '{"final": "gave up"}'])
    assert agent.run("t") == "gave up"
    assert "DENIED" in agent.llm.prompts[-1]


def test_agent_run_respects_step_limit(tmp_path):
    agent = loop_agent(tmp_path, ['{"tool": "read_file", "args": {"path": "x"}}'] * 3)
    result = agent.run("t", max_steps=3)
    assert "3-step limit" in result
    assert agent.trace[-1]["kind"] == "timeout"


def test_agent_run_detects_repeated_calls(tmp_path):
    agent = loop_agent(tmp_path, [
        '{"tool": "read_file", "args": {"path": "data/a.txt"}}',
        '{"tool": "read_file", "args": {"path": "data/a.txt"}}',
        '{"final": "done"}',
    ])
    assert agent.run("t") == "done"
    actions = [entry for entry in agent.trace if entry["kind"] == "action"]
    assert len(actions) == 1
    assert "already called" in agent.llm.prompts[-1]


def test_agent_run_survives_tool_type_error(tmp_path):
    charter = Charter(role="r", description="d", allowed_tools={"read_file"})
    agent = make_agent(tmp_path, charter)
    agent.tools["read_file"] = Tool(
        "read_file", "read", lambda path, extra=0: "ok", SAFE, path_arg="path",
    )
    agent.llm = FakeLLM([
        '{"tool": "read_file", "args": {}}',
        '{"final": "recovered from tool error"}',
    ])
    assert agent.run("t") == "recovered from tool error"
    assert "TOOL ERROR" in agent.llm.prompts[-1]


def test_agent_run_without_llm(tmp_path):
    charter = Charter(role="r", description="d", allowed_tools={"read_file"})
    agent = make_agent(tmp_path, charter)
    with pytest.raises(AgentError, match="no LLM"):
        agent.run("t")


# ---- tools -----------------------------------------------------------------

def test_build_tools_names():
    tools = build_tools(out_dir="data")
    for name in ("list_videos", "get_video", "ask_video", "read_file", "list_dir",
                 "search_web", "summarize_video", "write_file", "edit_code",
                 "run_command", "download_model"):
        assert name in tools


def test_run_command_allowlist():
    assert "not allowlisted" in _run_command("rm -rf /")
    assert "not allowlisted" in _run_command("curl http://x")


def test_read_file_missing():
    assert _read_file("/nonexistent/path") == "not found: /nonexistent/path"


# ---- registry --------------------------------------------------------------

def test_build_agent_unknown():
    with pytest.raises(KeyError, match="unknown agent"):
        build_agent("wizard", AgentOptions(out_dir="data"))


def test_analyst_has_no_write_tools(tmp_path):
    options = AgentOptions(out_dir=str(tmp_path), llm=FakeLLM([]), model="m")
    agent = make_analyst(options)
    assert "write_file" not in agent.tools
    assert "summarize_video" not in agent.tools
    assert "list_videos" in agent.tools


def test_coder_has_write_tools(tmp_path):
    options = AgentOptions(out_dir=str(tmp_path), llm=FakeLLM([]), model="m")
    agent = make_coder(options)
    assert "write_file" in agent.tools
    assert "edit_code" in agent.tools
    assert "download_model" in agent.tools


def test_agent_options_wire_llm_and_gate(tmp_path):
    fake = FakeLLM([])
    options = AgentOptions(out_dir=str(tmp_path), llm=fake, model="m")
    agent = make_analyst(options)
    assert agent.llm is fake
    assert agent.model == "m"
    assert options.gate.audit_path == tmp_path / "agents" / "audit.jsonl"
