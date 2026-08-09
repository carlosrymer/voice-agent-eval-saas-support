"""The simulated caller: a text brain driving a synthesised voice.

The caller is deliberately *not* a realtime model. It is the same
text-simulator design tau2 uses -- persona plus scenario plus the tau2 voice
guidelines, with the customer-side tools attached -- and its output is then
spoken by TTS. Two reasons.

First, it holds the caller constant between the text baseline and this voice
arm. The published text results for these 16 tasks were produced by a
`gemini-3.6-flash` text user simulator following the same guidelines. Swapping
in a realtime caller here would change both sides of the experiment at once,
and any voice-versus-text difference could be the caller rather than the agent.

Second, it makes the caller's words *known*. Because the harness authors every
utterance before speaking it, caller word error rate is measurable against an
exact reference, and the whole transcript-versus-audio judge comparison has a
ground truth for at least one speaker.

Barge-in is scripted rather than emergent, for the same reason: an interruption
that happens spontaneously is not reproducible, and a barge-in metric computed
over an uncontrolled number of interruptions is not comparable between arms.
:class:`BargeInPlan` says exactly which turns interrupt and how far into the
agent's speech, and the plan is recorded with the results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tau2.data_model.tasks import Task

from voiceval.domain import execute_user_tool, user_tool_specs
from voiceval.llm import DEFAULT_TEXT_MODEL, GeminiClient, model_content, user_content

GUIDELINES_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "tau2_data"
    / "tau2"
    / "user_simulator"
    / "simulation_guidelines_voice_tools.md"
)

STOP = "###STOP###"
TRANSFER = "###TRANSFER###"
OUT_OF_SCOPE = "###OUT-OF-SCOPE###"
TERMINATORS = (STOP, TRANSFER, OUT_OF_SCOPE)


@dataclass
class BargeInPlan:
    """Which agent turns the caller interrupts, and how far in.

    ``turns`` are zero-based agent turn indices. ``offset_s`` is how long after
    the agent starts speaking the caller begins. An interruption is only
    attempted if the agent is actually still talking at that point; ones that
    miss are counted and reported so a run cannot quietly test barge-in zero
    times while reporting a yield rate.
    """

    turns: tuple[int, ...] = ()
    offset_s: float = 1.0
    utterance: str = "Sorry, hang on a second."

    def fires_on(self, turn_index: int) -> bool:
        return turn_index in self.turns


@dataclass
class CallerAction:
    kind: str  # "speak" | "tool" | "end"
    text: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    terminator: str | None = None


def scenario_brief(task: Task) -> str:
    us = task.user_scenario
    ins = us.instructions
    bits = [f"## Your persona\n\n{us.persona or 'A reasonable customer.'}", "\n## Your scenario\n"]
    for label, value in (
        ("Why you are calling", getattr(ins, "reason_for_call", None)),
        ("What you know", getattr(ins, "known_info", None)),
        ("What you do NOT know", getattr(ins, "unknown_info", None)),
        ("What you need to do on this call", getattr(ins, "task_instructions", None)),
    ):
        if value:
            bits.append(f"**{label}:** {value}\n")
    return "\n".join(bits)


class CallerSimulator:
    def __init__(
        self,
        task: Task,
        env,
        client: GeminiClient,
        model: str = DEFAULT_TEXT_MODEL,
        temperature: float = 1.0,
        max_tool_hops: int = 4,
    ):
        self.task = task
        self.env = env
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tool_hops = max_tool_hops
        self.guidelines = GUIDELINES_PATH.read_text()
        self.tools = [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in user_tool_specs(env)
        ]
        self.history: list[dict[str, Any]] = []
        self.tool_calls_made: list[tuple[str, dict[str, Any], str]] = []
        self.ended: str | None = None

    @property
    def system_instruction(self) -> str:
        return f"{self.guidelines}\n\n{scenario_brief(self.task)}"

    def observe_agent(self, text: str) -> None:
        """Record what the support agent said. This is *input* to the caller."""
        if text:
            self.history.append(user_content([{"text": text}]))

    def _append_caller(self, text: str) -> None:
        """Record what the caller said.

        The simulator IS the caller, so the caller's own lines are the `model`
        turns and the support agent's lines are the `user` turns. Getting this
        backwards does not error -- it produces a model that fluently plays the
        *other* side of the call, inventing both halves of a conversation that
        never happened. That failure is invisible in the token counts and
        obvious in the transcript, which is why the smoke test prints one.
        """
        self.history.append(model_content([{"text": text}]))

    def next_action(self) -> CallerAction:
        """One caller step: possibly some tool use, then something to say."""
        if self.ended:
            return CallerAction("end", terminator=self.ended)

        for _ in range(self.max_tool_hops):
            contents = _alternate(self.history) or [user_content([{"text": "(the line connects)"}])]
            resp = self.client.generate(
                self.model,
                contents,
                system_instruction=self.system_instruction,
                tools=self.tools or None,
                temperature=self.temperature,
                max_output_tokens=400,
            )
            if resp.function_calls:
                fc = resp.function_calls[0]
                result = execute_user_tool(self.env, fc["name"], fc["args"])
                self.tool_calls_made.append((fc["name"], fc["args"], result.content))
                # Feed the result back as narration; the customer-side tools are
                # things the caller does in their own workspace, not speech.
                self.history.append(
                    user_content([{"text": f"[you used {fc['name']}; it returned: {result.content}]"}])
                )
                continue

            text = resp.text.strip()
            term = next((t for t in TERMINATORS if t in text), None)
            if term:
                self.ended = term
                cleaned = text.replace(term, "").strip()
                if cleaned:
                    self._append_caller(cleaned)
                    return CallerAction("speak", text=cleaned, terminator=term)
                return CallerAction("end", terminator=term)

            text = _strip_stage_directions(text)
            if not text:
                text = "Sorry, could you say that again?"
            self._append_caller(text)
            return CallerAction("speak", text=text)

        return CallerAction("end", terminator=OUT_OF_SCOPE)


def _alternate(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive same-role turns so the API accepts the history."""
    out: list[dict[str, Any]] = []
    for item in history:
        if out and out[-1]["role"] == item["role"]:
            out[-1] = {
                "role": item["role"],
                "parts": [
                    {
                        "text": " ".join(
                            p.get("text", "") for p in out[-1]["parts"] + item["parts"]
                        ).strip()
                    }
                ],
            }
        else:
            out.append({"role": item["role"], "parts": list(item["parts"])})
    if out and out[0]["role"] != "user":
        out.insert(0, user_content([{"text": "(the line connects)"}]))
    if out and out[-1]["role"] == "model":
        # The API rejects a request whose last turn is the model's. This
        # happens whenever the agent said nothing back -- which on a phone call
        # is itself information, so it is passed through as what it is rather
        # than papered over. The tau2 voice guidelines tell the caller how to
        # react to silence, and this is what triggers that behaviour.
        out.append(user_content([{"text": "(silence on the line)"}]))
    return out


_STAGE = re.compile(r"^\s*[\[(][^\])]{0,80}[\])]\s*")


def _strip_stage_directions(text: str) -> str:
    """Remove a leading '(sighs)' style direction that TTS would read aloud."""
    prev = None
    while prev != text:
        prev = text
        text = _STAGE.sub("", text).strip()
    return text.replace("[pause]", "...").strip()
