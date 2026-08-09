"""Bridge from the tau2 Loopline support domain to the voice harness.

The whole point of reusing this domain is that a text baseline for it already
exists and is public: the same 16 tasks, the same 7-rule policy, the same 18
agent tools, scored by the same code. If I re-implemented scoring here, the
voice-versus-text comparison would be comparing two graders as much as two
channels. So Outcome and the action half of Execution are computed by tau2's
own :class:`EnvironmentEvaluator` and :class:`ActionEvaluator`, fed a message
trajectory reconstructed from the call. Only the policy auditor and the
Experience metrics are mine, and the policy auditor is ported rather than
rewritten.

The one thing that genuinely has to change for voice is the system prompt. The
text policy says "You are talking to a customer over chat"; on a phone call that
is false, and leaving it would test the model's ability to ignore its own
instructions rather than its ability to follow them. :func:`voice_system_prompt`
appends a channel note and nothing else -- no extra guidance, no hints about the
rules, nothing that would make the voice arm easier or harder than the text arm
on the substance being measured. The exact appended text is in this file so
anyone can check that claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import saas_support  # noqa: F401  isort:skip  (sets TAU2_DATA_DIR before tau2 loads)

from tau2.data_model.message import (  # noqa: E402
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import Task  # noqa: E402
from tau2.environment.environment import Environment  # noqa: E402

from saas_support.environment import get_environment, get_tasks  # noqa: E402
from saas_support.register import register_domain  # noqa: E402
from saas_support.utils import DOMAIN_NAME, SAAS_POLICY_PATH  # noqa: E402

from voiceval.providers.base import ToolSpec  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Appended verbatim to the text policy for the voice arm. Deliberately minimal:
#: it corrects the channel and the spelled-out-identifier problem that only
#: exists in speech, and says nothing about any of the seven rules.
VOICE_CHANNEL_NOTE = """

## Channel

You are on a **live phone call**, not a chat. Everything you say is spoken aloud
and everything the customer says is transcribed from speech.

- Speak in short, natural sentences. Do not read out markdown, bullet lists,
  URLs or raw JSON.
- Identifiers are heard, not seen. When you read one out, say it in groups
  ("acct one zero four two"). When the customer gives you one, read it back to
  confirm before you use it.
- The customer can interrupt you. If they start speaking, stop and listen.
- Do not ask the customer to click, paste, or look at anything on this call
  unless they can plainly do it while talking to you.
"""


def ensure_registered() -> None:
    register_domain()


def load_policy(voice: bool = True) -> str:
    text = Path(SAAS_POLICY_PATH).read_text()
    return text + VOICE_CHANNEL_NOTE if voice else text


def voice_system_prompt() -> str:
    return load_policy(voice=True)


def tasks(split: str | None = None) -> list[Task]:
    ensure_registered()
    return get_tasks(split) if split else get_tasks(None)


def task_by_id(task_id: str) -> Task:
    for t in tasks():
        if t.id == task_id:
            return t
    raise KeyError(f"unknown task {task_id!r}")


def new_environment(task: Task | None = None) -> Environment:
    """A fresh environment, with the task's initial state applied."""
    env = get_environment()
    if task is not None and task.initial_state is not None:
        env.set_state(
            initialization_data=task.initial_state.initialization_data,
            initialization_actions=task.initial_state.initialization_actions,
            message_history=[],
        )
    return env


def _clean_schema(schema: Any) -> Any:
    """Strip JSON-Schema keys the Gemini function-declaration parser rejects.

    ``title`` and ``default`` come from pydantic and are informational only.
    Sending them gets the whole setup frame rejected, which presents as a
    mysterious connection failure rather than a schema error.
    """
    if isinstance(schema, dict):
        return {
            k: _clean_schema(v)
            for k, v in schema.items()
            if k not in {"title", "default", "additionalProperties"}
        }
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]
    return schema


def agent_tool_specs(env: Environment) -> list[ToolSpec]:
    out: list[ToolSpec] = []
    for tool in env.get_tools():
        fn = tool.openai_schema["function"]
        params = _clean_schema(fn.get("parameters") or {"type": "object", "properties": {}})
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        out.append(ToolSpec(name=fn["name"], description=fn.get("description", ""),
                            parameters=params))
    return out


def user_tool_specs(env: Environment) -> list[ToolSpec]:
    out: list[ToolSpec] = []
    for tool in env.get_user_tools():
        fn = tool.openai_schema["function"]
        params = _clean_schema(fn.get("parameters") or {"type": "object", "properties": {}})
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        out.append(ToolSpec(name=fn["name"], description=fn.get("description", ""),
                            parameters=params))
    return out


@dataclass
class ToolResult:
    ok: bool
    content: str
    error: str | None = None


def execute_agent_tool(env: Environment, name: str, args: dict[str, Any]) -> ToolResult:
    return _execute(env, name, args, requestor="assistant")


def execute_user_tool(env: Environment, name: str, args: dict[str, Any]) -> ToolResult:
    return _execute(env, name, args, requestor="user")


def _execute(env: Environment, name: str, args: dict[str, Any], requestor: str) -> ToolResult:
    try:
        result = env.make_tool_call(name, requestor=requestor, **(args or {}))
        return ToolResult(True, _stringify(result))
    except Exception as exc:
        # A tool error is a legitimate outcome, not a harness failure: the agent
        # is supposed to cope with "that account does not exist". It is recorded
        # and handed back to the model exactly as tau2's text runner does.
        return ToolResult(False, f"Error: {exc}", error=f"{type(exc).__name__}: {exc}")


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        if hasattr(result, "model_dump"):
            return json.dumps(result.model_dump(), default=str)
        return json.dumps(result, default=str)
    except Exception:
        return str(result)


# --------------------------------------------------------------------------
# Trajectory reconstruction
# --------------------------------------------------------------------------
def build_trajectory(record) -> list[Message]:
    """Rebuild a tau2 message trajectory from a voice call.

    tau2's evaluators think in messages, and this domain's text baseline was
    scored from exactly such a list. Rebuilding one from the call means the
    voice arm goes through the identical evaluator, so a difference in the score
    is a difference in the agent's behaviour rather than in how it was graded.

    Blocks are ordered by when they happened, but **a tool call and its result
    are emitted adjacently**, never interleaved by timestamp. tau2 rejects a
    trajectory where a tool message does not directly follow its call, and on a
    voice call the timestamps genuinely do interleave -- the caller keeps talking
    while a tool runs. Sorting purely by time therefore produced a trajectory the
    evaluator refused, which surfaced as every Outcome score coming back "not
    scored" rather than as a wrong number.
    """
    from voiceval.metrics.timeline import CallRecord  # local import, avoids a cycle

    assert isinstance(record, CallRecord)

    blocks: list[tuple[float, int, list[Message]]] = []
    order = 0

    def add(t: float, msgs: list[Message]) -> None:
        nonlocal order
        order += 1
        blocks.append((t, order, msgs))

    for u in record.caller_utterances:
        add(u.start_t, [UserMessage(role="user", content=u.text, is_audio=True)])

    claimed: set[str] = set()

    for a in record.agent_utterances:
        t0 = a.turn_started_t if a.turn_started_t is not None else (a.audio_start_t or 0.0)
        t1 = a.audio_end_t if a.audio_end_t is not None else t0
        mine = [
            te
            for te in record.tool_executions
            if te.requestor == "assistant"
            and te.call_id not in claimed
            and t0 - 1e-6 <= te.requested_t <= t1 + 1e-6
        ]
        for te in mine:
            claimed.add(te.call_id)
            add(te.requested_t, _call_and_result(te, "assistant"))
        if a.text:
            add(t1, [AssistantMessage(role="assistant", content=a.text, is_audio=True)])

    # Anything outside a recognised utterance window, plus every customer-side
    # call, still has to appear or the evaluator will not see actions that
    # genuinely happened.
    for te in record.tool_executions:
        if te.call_id in claimed:
            continue
        claimed.add(te.call_id)
        add(te.requested_t, _call_and_result(te, te.requestor))

    blocks.sort(key=lambda b: (b[0], b[1]))
    out: list[Message] = []
    for _, _, msgs in blocks:
        out.extend(msgs)
    for i, m in enumerate(out):
        m.turn_idx = i
    return out


def _call_and_result(te, requestor: str) -> list[Message]:
    """One tool call plus its result, as an inseparable pair."""
    call = ToolCall(id=te.call_id, name=te.name, arguments=te.args, requestor=requestor)
    if requestor == "assistant":
        head: Message = AssistantMessage(
            role="assistant", content=None, is_audio=True, tool_calls=[call]
        )
    else:
        head = UserMessage(role="user", content=None, is_audio=True, tool_calls=[call])
    return [
        head,
        ToolMessage(
            id=te.call_id, role="tool", content=te.result,
            requestor=requestor, error=not te.ok,
        ),
    ]


def env_constructor(**kwargs):
    """Fresh environment for tau2's evaluators.

    Must accept keyword arguments: tau2 calls this with ``solo_mode`` and
    whatever ``env_kwargs`` it was given. A zero-argument version raised a
    TypeError inside the evaluator, which :func:`score_outcome` correctly
    reported as a harness error rather than a task failure -- so every Outcome
    score came back as "not scored" instead of silently becoming a zero. That is
    the failure mode the error handling was written for, and it is the reason
    the whole Outcome axis did not quietly read 0%.
    """
    return get_environment(**kwargs)


DOMAIN = DOMAIN_NAME
