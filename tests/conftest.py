"""Synthetic call construction with declared ground truth.

:class:`SyntheticCall` builds a real :class:`CallRecord` -- real rendered audio
on both tracks, real provider events, real tool-execution records -- from a
specification I write in the test. The metrics under test then run on exactly
the same objects they will see in a funded run, but against numbers I chose.

That is the whole point of this file: it means "end-of-turn latency is 620 ms"
can be *asserted* rather than observed, months before there is any credit to
spend on observing it.
"""

from __future__ import annotations

from voiceval.audio import fixtures as fx
from voiceval.audio.pcm import PCM, place
from voiceval.metrics.timeline import AgentUtterance, CallerUtterance, CallRecord, ToolExecution
from voiceval.providers.base import EventKind, ServerEvent

CALLER_RATE = 16000
AGENT_RATE = 24000

DEFAULT_CAPS = {
    "server_turn_detection": False,
    "server_barge_in": True,
    "emits_interrupt_event": True,
    "emits_caller_transcript": True,
    "emits_agent_transcript": True,
    "emits_turn_start": True,
    "input_sample_rate_hz": CALLER_RATE,
    "output_sample_rate_hz": AGENT_RATE,
    "native_tool_calling": True,
    "notes": "test fixture",
}


class SyntheticCall:
    def __init__(self, call_id: str = "c1", task_id: str = "T01", caps: dict | None = None):
        self.call_id = call_id
        self.task_id = task_id
        self.caps = dict(DEFAULT_CAPS if caps is None else caps)
        self._caller: list[tuple[float, PCM]] = []
        self._agent: list[tuple[float, PCM]] = []
        self.caller_utterances: list[CallerUtterance] = []
        self.agent_utterances: list[AgentUtterance] = []
        self.tools: list[ToolExecution] = []
        self.events: list[ServerEvent] = []
        self._seq = 0
        self.ended_reason = "completed"

    # -- events ------------------------------------------------------------
    def event(self, kind: EventKind, t: float, **kw) -> "SyntheticCall":
        self._seq += 1
        self.events.append(ServerEvent(kind=kind, t=t, seq=self._seq, **kw))
        return self

    # -- speech ------------------------------------------------------------
    def caller_says(
        self,
        at: float,
        duration: float,
        text: str = "hello",
        *,
        asr: str | None = None,
        asr_delay: float | None = 0.12,
        barge_in: bool = False,
        level_dbfs: float = -20.0,
    ) -> "SyntheticCall":
        pcm = fx.speech_like(duration, f0_hz=190.0, level_dbfs=level_dbfs, rate=CALLER_RATE, seed=11)
        self._caller.append((at, pcm))
        asr_t = None
        if asr is not None or asr_delay is not None:
            asr_t = at + duration + (asr_delay or 0.0)
            self.event(
                EventKind.CALLER_TRANSCRIPT,
                asr_t,
                text=asr if asr is not None else text,
                is_final=True,
            )
        self.caller_utterances.append(
            CallerUtterance(
                index=len(self.caller_utterances),
                text=text,
                start_t=at,
                end_t=at + duration,
                is_barge_in=barge_in,
                asr_text=asr if asr is not None else text,
                asr_final_t=asr_t,
            )
        )
        return self

    def agent_says(
        self,
        at: float,
        duration: float,
        text: str = "sure, one moment",
        *,
        turn_started_t: float | None = None,
        interrupted: bool = False,
        emit_events: bool = True,
        level_dbfs: float = -20.0,
    ) -> "SyntheticCall":
        pcm = fx.speech_like(duration, f0_hz=105.0, level_dbfs=level_dbfs, rate=AGENT_RATE, seed=22)
        self._agent.append((at, pcm))
        if emit_events:
            if turn_started_t is not None:
                self.event(EventKind.AGENT_TURN_STARTED, turn_started_t)
            self.event(EventKind.AGENT_TRANSCRIPT, at, text=text, is_final=True)
            if interrupted:
                self.event(EventKind.INTERRUPTED, at + duration, message="caller barge-in")
            else:
                self.event(EventKind.AGENT_TURN_COMPLETE, at + duration)
        self.agent_utterances.append(
            AgentUtterance(
                index=len(self.agent_utterances),
                text=text,
                audio_start_t=at,
                audio_end_t=at + duration,
                turn_started_t=turn_started_t if turn_started_t is not None else at,
                completed_t=at + duration,
                interrupted=interrupted,
            )
        )
        return self

    # -- tools -------------------------------------------------------------
    def tool(
        self,
        name: str,
        args: dict,
        requested_t: float,
        duration: float,
        *,
        result: str = "ok",
        requestor: str = "assistant",
        ok: bool = True,
    ) -> "SyntheticCall":
        cid = f"call_{len(self.tools)}"
        self.tools.append(
            ToolExecution(
                call_id=cid,
                name=name,
                args=args,
                requestor=requestor,
                requested_t=requested_t,
                started_t=requested_t,
                finished_t=requested_t + duration,
                ok=ok,
                result=result,
            )
        )
        self.event(EventKind.TOOL_CALL, requested_t, call_id=cid, tool_name=name, tool_args=args)
        return self

    def dropped_tool(self, name: str, args: dict, requested_t: float) -> "SyntheticCall":
        """A tool call that was requested and never completed."""
        cid = f"call_{len(self.tools)}"
        self.tools.append(
            ToolExecution(
                call_id=cid,
                name=name,
                args=args,
                requested_t=requested_t,
                started_t=requested_t,
                finished_t=requested_t,  # finished <= started marks it as never run
                ok=False,
                result="",
                error="never completed",
            )
        )
        self.event(EventKind.TOOL_CALL, requested_t, call_id=cid, tool_name=name, tool_args=args)
        return self

    # -- build -------------------------------------------------------------
    def build(self, total_s: float | None = None, noise_dbfs: float = -62.0) -> CallRecord:
        end = total_s or (
            max(
                [at + p.duration_s for at, p in self._caller + self._agent] or [0.0],
            )
            + 0.5
        )
        caller = fx.noise(end, noise_dbfs, CALLER_RATE, seed=3)
        for at, p in self._caller:
            caller = place(caller, p, at)
        agent = fx.noise(end, noise_dbfs, AGENT_RATE, seed=4)
        for at, p in self._agent:
            agent = place(agent, p, at)

        self.events.sort(key=lambda e: (e.t, e.seq))
        return CallRecord(
            call_id=self.call_id,
            task_id=self.task_id,
            trial=0,
            provider="fixture",
            model="fixture",
            synthetic=True,
            capabilities=self.caps,
            events=self.events,
            tool_executions=self.tools,
            caller_utterances=self.caller_utterances,
            agent_utterances=self.agent_utterances,
            caller_track=caller,
            agent_track=agent,
            duration_s=end,
            ended_reason=self.ended_reason,
        )
