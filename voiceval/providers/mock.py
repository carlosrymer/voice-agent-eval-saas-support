"""A deterministic offline realtime provider.

This is not a stub. It is a small simulated realtime server that implements the
full :class:`VoiceSession` contract -- streaming audio out in chunks, requesting
tool calls, waiting for results, honouring barge-in and cancellation -- against
a script and a virtual clock. Two things follow from that.

First, the whole harness runs end to end with no API key and no spend: the
orchestrator, the tool executor against the real tau2 environment, the action
ledger, every Experience metric, the report generator and the site all execute
on real data produced by real code paths. Only the model is simulated.

Second, and more importantly, the *measurement* layer can be tested against
known truth. When the script says "respond 620 ms after the caller stops, then
speak for 3.1 s, and yield 180 ms after being interrupted", the metrics have a
right answer to be checked against. A latency decomposition that has only ever
seen real calls has never been checked against anything.

Virtual time is owned by this session. Events carry their scripted timestamps;
the clock is advanced to each event as it is yielded. A caller that sleeps on
the same clock to schedule a barge-in therefore interleaves deterministically
with audio playback, with no wall-clock waiting and no scheduler jitter.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from voiceval.audio import fixtures as fx
from voiceval.audio.pcm import PCM
from voiceval.providers.base import (
    Clock,
    EventKind,
    ProviderCapabilities,
    ServerEvent,
    SessionConfig,
    VirtualClock,
    VoiceProvider,
    VoiceSession,
    register_provider,
)

CHUNK_MS = 20.0
MOCK_INPUT_RATE = 16000
MOCK_OUTPUT_RATE = 24000


@dataclass(frozen=True)
class MockToolCall:
    name: str
    args: dict[str, Any]
    #: Delay from turn start (or from the previous tool result) before this
    #: call is requested.
    delay_s: float = 0.15


@dataclass
class MockTurn:
    """Scripted agent behaviour in response to one caller turn."""

    text: str = "Thanks, one moment."
    #: Caller-stop to first sign of response. The dominant term in end-of-turn latency.
    response_delay_s: float = 0.62
    #: Turn-start to first audio byte, when no tool call intervenes.
    ttfa_after_start_s: float = 0.18
    tool_calls: list[MockToolCall] = field(default_factory=list)
    #: Last tool result to first audio byte.
    post_tool_delay_s: float = 0.34
    speech_duration_s: float = 2.4
    #: How long after caller onset this turn's audio actually stops, if interrupted.
    yield_latency_s: float = 0.18
    #: Emit an explicit INTERRUPTED frame when barged into.
    signals_interrupt: bool = True
    #: What the provider's ASR claims the caller said. None => echo ground truth.
    asr_text: str | None = None
    asr_delay_s: float = 0.12
    f0_hz: float = 105.0


@dataclass
class MockScript:
    turns: list[MockTurn] = field(default_factory=list)
    #: Used once the scripted turns run out, so a longer conversation than the
    #: script does not crash -- it just gets a generic turn.
    default_turn: MockTurn = field(default_factory=MockTurn)
    error_on_turn: int | None = None

    def turn(self, i: int) -> MockTurn:
        return self.turns[i] if i < len(self.turns) else self.default_turn


class MockVoiceSession(VoiceSession):
    def __init__(self, provider: "MockVoiceProvider", config: SessionConfig, clock: Clock):
        self.provider = provider
        self.config = config
        self.clock = clock
        self.capabilities = provider.capabilities()
        self.script = provider.script
        self._seq = 0
        self._turn_index = -1
        #: Scheduled but not yet yielded events, ordered by time.
        self._plan: list[ServerEvent] = []
        self._closed = False
        self._pending_tools: dict[str, MockToolCall] = {}
        self._last_ground_truth: str | None = None
        self._interrupt_count = 0
        #: (first_audio_t, audio_end_t) for the turn currently being spoken.
        self._speech_span: tuple[float, float] | None = None
        self._emit(EventKind.SESSION_OPENED, self.clock.now())

    # -- internals ---------------------------------------------------------
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _emit(self, kind: EventKind, t: float, **kw: Any) -> ServerEvent:
        ev = ServerEvent(kind=kind, t=round(t, 6), seq=self._next_seq(), **kw)
        self._plan.append(ev)
        self._plan.sort(key=lambda e: (e.t, e.seq))
        return ev

    def _speech_end_of_plan(self) -> float:
        audio = [e for e in self._plan if e.kind == EventKind.AGENT_AUDIO]
        if not audio:
            return self.clock.now()
        last = audio[-1]
        return last.t + (last.audio.duration_s if last.audio else 0.0)

    # -- VoiceSession ------------------------------------------------------
    async def send_audio(self, pcm: PCM, *, ground_truth_text: str | None = None) -> None:
        """Accept caller audio.

        If the agent is mid-utterance when this lands, this is a barge-in: the
        remaining planned audio is truncated at ``yield_latency_s`` after the
        caller's onset, exactly as a server-side barge-in implementation would
        cut its own output stream.
        """
        if self._closed:
            raise RuntimeError("session is closed")
        now = self.clock.now()
        if ground_truth_text is not None:
            self._last_ground_truth = ground_truth_text

        speaking = (
            self._speech_span is not None
            and self._speech_span[0] <= now < self._speech_span[1]
        )
        if speaking and self.capabilities.server_barge_in:
            await self._do_barge_in(now)

    async def _do_barge_in(self, caller_onset_t: float) -> None:
        turn = self.script.turn(max(0, self._turn_index))
        cut_at = caller_onset_t + turn.yield_latency_s
        kept: list[ServerEvent] = []
        for e in self._plan:
            if e.kind == EventKind.AGENT_AUDIO and e.t >= cut_at:
                continue
            if e.kind == EventKind.AGENT_TURN_COMPLETE:
                continue
            kept.append(e)
        self._plan = kept
        if self._speech_span is not None:
            self._speech_span = (self._speech_span[0], min(self._speech_span[1], cut_at))
        self._interrupt_count += 1
        if turn.signals_interrupt and self.capabilities.emits_interrupt_event:
            self._emit(EventKind.INTERRUPTED, cut_at, message="caller barge-in")
        else:
            # Providers without an interrupt frame still stop; the turn simply
            # ends. Barge-in then has to be measured from the audio.
            self._emit(EventKind.AGENT_TURN_COMPLETE, cut_at)

    async def commit_turn(self) -> None:
        """Caller finished speaking: plan the agent's whole response."""
        if self._closed:
            raise RuntimeError("session is closed")
        self._turn_index += 1
        i = self._turn_index
        turn = self.script.turn(i)
        t = self.clock.now()
        self._speech_span = None

        if turn.asr_text is not None or self._last_ground_truth is not None:
            self._emit(
                EventKind.CALLER_TRANSCRIPT,
                t + turn.asr_delay_s,
                text=turn.asr_text if turn.asr_text is not None else self._last_ground_truth,
                is_final=True,
            )
        self._last_ground_truth = None

        if self.script.error_on_turn == i:
            self._emit(EventKind.ERROR, t + 0.1, message="simulated provider error")
            return

        start_t = t + turn.response_delay_s
        self._emit(EventKind.AGENT_TURN_STARTED, start_t)

        if turn.tool_calls:
            cursor = start_t
            for n, tc in enumerate(turn.tool_calls):
                cursor += tc.delay_s
                call_id = f"mock_call_{i}_{n}"
                self._pending_tools[call_id] = tc
                self._emit(
                    EventKind.TOOL_CALL,
                    cursor,
                    call_id=call_id,
                    tool_name=tc.name,
                    tool_args=dict(tc.args),
                )
            # Audio is planned once the last tool result comes back, in
            # send_tool_result. A realtime model cannot speak the answer before
            # it has the answer, and a harness that plans it up front would
            # measure a tool latency of zero.
            return

        self._plan_speech(start_t + turn.ttfa_after_start_s, turn)

    def _plan_speech(self, first_audio_t: float, turn: MockTurn) -> None:
        audio = fx.speech_like(
            turn.speech_duration_s, f0_hz=turn.f0_hz, rate=MOCK_OUTPUT_RATE, seed=1234
        )
        chunk_s = CHUNK_MS / 1000.0
        n_chunks = max(1, int(round(turn.speech_duration_s / chunk_s)))
        for c in range(n_chunks):
            self._emit(
                EventKind.AGENT_AUDIO,
                first_audio_t + c * chunk_s,
                audio=audio.slice_s(c * chunk_s, (c + 1) * chunk_s),
            )
        if turn.text:
            self._emit(EventKind.AGENT_TRANSCRIPT, first_audio_t, text=turn.text, is_final=True)
        self._emit(EventKind.AGENT_TURN_COMPLETE, first_audio_t + turn.speech_duration_s)
        self._speech_span = (first_audio_t, first_audio_t + turn.speech_duration_s)

    async def send_tool_result(self, call_id: str, name: str, payload: Any) -> None:
        if call_id not in self._pending_tools:
            raise KeyError(f"unknown tool call id {call_id!r}")
        del self._pending_tools[call_id]
        if self._pending_tools:
            return  # wait for the rest of this turn's calls
        turn = self.script.turn(self._turn_index)
        _ = json.dumps(payload, default=str)  # payload must be serialisable
        self._plan_speech(self.clock.now() + turn.post_tool_delay_s, turn)

    async def cancel_response(self) -> None:
        await self._do_barge_in(self.clock.now())

    async def events(self) -> AsyncIterator[ServerEvent]:
        while self._plan and not self._closed:
            ev = self._plan[0]
            # Honour the scripted time. On a virtual clock that means jumping to
            # it for free; on a wall clock it means actually waiting, so that a
            # mock-driven call has the same temporal shape a real one does and
            # things that depend on timing -- the playout drain, a scheduled
            # barge-in -- get a chance to happen. Delivering the whole plan
            # instantly made every scripted interruption arrive after the turn
            # it was meant to interrupt had already ended.
            if ev.t > self.clock.now():
                if isinstance(self.clock, VirtualClock):
                    self.clock.set(ev.t)
                else:
                    await asyncio.sleep(min(ev.t - self.clock.now(), 30.0))
            self._plan.pop(0)
            yield ev
            if ev.kind in (
                EventKind.AGENT_TURN_COMPLETE,
                EventKind.INTERRUPTED,
                EventKind.ERROR,
            ):
                return
            if ev.kind == EventKind.TOOL_CALL:
                return  # orchestrator must supply a result before more happens

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._plan = []


@register_provider
class MockVoiceProvider(VoiceProvider):
    """Offline provider. Every number it produces is synthetic by construction."""

    name = "mock"
    wire_verified = True  # there is no wire; the contract is exercised in tests

    def __init__(
        self,
        script: MockScript | None = None,
        *,
        server_barge_in: bool = True,
        emits_interrupt_event: bool = True,
        emits_turn_start: bool = True,
    ):
        self.script = script or MockScript()
        self._caps = ProviderCapabilities(
            server_turn_detection=False,
            server_barge_in=server_barge_in,
            emits_interrupt_event=emits_interrupt_event,
            emits_caller_transcript=True,
            emits_agent_transcript=True,
            emits_turn_start=emits_turn_start,
            input_sample_rate_hz=MOCK_INPUT_RATE,
            output_sample_rate_hz=MOCK_OUTPUT_RATE,
            notes="Offline simulator on a virtual clock. Results are synthetic.",
        )

    def capabilities(self) -> ProviderCapabilities:
        return self._caps

    async def connect(self, config: SessionConfig, clock: Clock | None = None) -> VoiceSession:
        return MockVoiceSession(self, config, clock or VirtualClock())
