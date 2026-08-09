"""The provider seam: one normalized voice-session contract, many vendors.

Everything above this module -- the orchestrator, the action ledger, every
Experience metric, both judges, the report -- is written against the types in
this file and has no idea which vendor produced a call. That is not architecture
for its own sake. Gemini Live is the only realtime stack I currently have a key
for, and a second one (OpenAI Realtime) is expected later; if the harness were
written against Gemini's wire format, the second stack would arrive as a rewrite
of the measurement layer, and every number measured before it would become
incomparable with every number measured after.

Three decisions carry most of the weight.

**Timestamps are stamped by the transport, not by the vendor.** Every
:class:`ServerEvent` carries ``t``, seconds since the session opened, recorded
the instant the frame came off the socket by :class:`Clock`. Vendors report
their own timings inconsistently or not at all, and a latency comparison across
vendors that trusts vendor-reported timings is measuring their telemetry, not
their speed. The one thing every vendor does identically is put bytes on a
socket, so that is what gets clocked.

**Capabilities are declared, not assumed.** Realtime stacks differ in ways that
change what a metric even means: whether the server does turn detection, whether
it cancels its own audio on barge-in, whether it emits an explicit interrupt
signal, what sample rates it speaks. :class:`ProviderCapabilities` makes those
differences data. A metric that cannot be computed for a given provider reports
``None`` and says why, rather than silently reporting a number that means
something different from the one next to it.

**Audio is PCM in, PCM out, rate attached.** No provider-specific containers
cross this boundary. Base64, protobuf, opus, whatever the vendor does is the
adapter's problem.

To add a provider, implement :class:`VoiceProvider` and :class:`VoiceSession`,
declare capabilities, and translate the vendor's frames into ``ServerEvent``.
That is the whole contract. See ``PROVIDERS.md`` for the worked mapping.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from voiceval.audio.pcm import PCM


# --------------------------------------------------------------------------
# Clock
# --------------------------------------------------------------------------
class Clock(ABC):
    """Source of session-relative time.

    Abstracted so the mock provider can run a 90-second call in milliseconds of
    virtual time while producing byte-identical timelines to a wall-clock run.
    Without this, every timing test would have to actually sleep, and the test
    suite would be too slow to run on every change -- which is the same as not
    having one.
    """

    @abstractmethod
    def now(self) -> float:
        """Seconds since the session opened."""

    @abstractmethod
    async def sleep(self, seconds: float) -> None: ...


class WallClock(Clock):
    def __init__(self) -> None:
        self._t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._t0

    async def sleep(self, seconds: float) -> None:
        import asyncio

        await asyncio.sleep(seconds)


class PausableWallClock(Clock):
    """Wall clock that can be stopped while the harness itself is thinking.

    The simulated caller needs a text-model call and a TTS round trip before it
    can say anything -- often ten seconds. That delay belongs to my test rig,
    not to the conversation: a real caller answers in about a second, and
    leaving the rig's latency in the timeline would invent ten-second silences
    that the friction metrics would then dutifully report as dead air, and would
    stretch every call duration into nonsense.

    So the clock is paused around caller-brain and speech-synthesis work. It is
    only ever paused when the provider has finished its turn and no audio is in
    flight, so nothing real is happening while time is stopped. Total paused
    time is recorded and published with the results rather than discarded.
    """

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self._paused_at: float | None = None
        self.total_paused_s: float = 0.0

    def now(self) -> float:
        if self._paused_at is not None:
            return self._paused_at - self._t0 - self.total_paused_s
        return time.monotonic() - self._t0 - self.total_paused_s

    def pause(self) -> None:
        if self._paused_at is None:
            self._paused_at = time.monotonic()

    def resume(self) -> None:
        if self._paused_at is not None:
            self.total_paused_s += time.monotonic() - self._paused_at
            self._paused_at = None

    async def sleep(self, seconds: float) -> None:
        import asyncio

        await asyncio.sleep(seconds)


class VirtualClock(Clock):
    """Time advances only when something asks it to. No real waiting."""

    def __init__(self) -> None:
        self._t = 0.0

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("virtual time does not go backwards")
        self._t += seconds

    def set(self, t: float) -> None:
        if t < self._t:
            raise ValueError(f"virtual time does not go backwards ({t} < {self._t})")
        self._t = t

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------
class EventKind(str, Enum):
    SESSION_OPENED = "session_opened"
    #: Provider began producing a response (first sign of life for this turn).
    AGENT_TURN_STARTED = "agent_turn_started"
    AGENT_AUDIO = "agent_audio"
    AGENT_TRANSCRIPT = "agent_transcript"
    #: The provider's ASR of what the caller said.
    CALLER_TRANSCRIPT = "caller_transcript"
    #: Provider's own end-of-caller-turn detection, if it has one.
    CALLER_SPEECH_STARTED = "caller_speech_started"
    CALLER_SPEECH_STOPPED = "caller_speech_stopped"
    TOOL_CALL = "tool_call"
    AGENT_TURN_COMPLETE = "agent_turn_complete"
    #: The provider abandoned the in-flight response, typically due to barge-in.
    INTERRUPTED = "interrupted"
    ERROR = "error"
    SESSION_CLOSED = "session_closed"


@dataclass(frozen=True)
class ServerEvent:
    """One normalized frame from the provider.

    ``t`` is session-relative seconds, stamped on receipt. ``raw`` keeps the
    vendor frame (minus audio bytes) so a trace can be audited back to the wire
    without the normalization layer being the only account of what happened.
    """

    kind: EventKind
    t: float
    seq: int = 0
    audio: PCM | None = None
    text: str | None = None
    is_final: bool = False
    call_id: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    message: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind.value, "t": round(self.t, 6), "seq": self.seq}
        if self.audio is not None:
            d["audio_ms"] = round(self.audio.duration_s * 1000.0, 3)
            d["audio_rate_hz"] = self.audio.sample_rate_hz
        for k in ("text", "call_id", "tool_name", "message"):
            v = getattr(self, k)
            if v:
                d[k] = v
        if self.tool_args:
            d["tool_args"] = self.tool_args
        if self.is_final:
            d["is_final"] = True
        return d


# --------------------------------------------------------------------------
# Session configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolSpec:
    """A tool offered to the voice agent, in vendor-neutral JSON-Schema form."""

    name: str
    description: str
    parameters: dict[str, Any]


class TurnDetection(str, Enum):
    #: The provider decides when the caller has stopped talking.
    SERVER_VAD = "server_vad"
    #: The harness decides and sends an explicit end-of-turn signal.
    CLIENT_COMMIT = "client_commit"


@dataclass(frozen=True)
class SessionConfig:
    system_instruction: str
    tools: tuple[ToolSpec, ...] = ()
    voice: str = "default"
    model: str = ""
    turn_detection: TurnDetection = TurnDetection.SERVER_VAD
    #: Ask the provider for text transcripts alongside audio where supported.
    request_input_transcript: bool = True
    request_output_transcript: bool = True
    temperature: float | None = None
    max_output_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider can actually do, so metrics know what they may claim."""

    #: Provider detects end-of-caller-turn itself.
    server_turn_detection: bool
    #: Provider stops its own audio when the caller starts talking.
    server_barge_in: bool
    #: Provider emits an explicit signal when it does so. Without this, barge-in
    #: yield latency has to be measured from the audio, which is still valid but
    #: is a different measurement and is labelled as such.
    emits_interrupt_event: bool
    emits_caller_transcript: bool
    emits_agent_transcript: bool
    #: Provider emits a distinguishable "response started" frame before the first
    #: audio byte. Where false, time-to-first-audio is the only available
    #: response-onset measure and inference/TTS cannot be separated.
    emits_turn_start: bool
    input_sample_rate_hz: int
    output_sample_rate_hz: int
    input_mime: str = "audio/pcm;rate=16000"
    output_mime: str = "audio/pcm;rate=24000"
    native_tool_calling: bool = True
    #: Free-text notes surfaced in the report next to any provider comparison.
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            k: getattr(self, k)
            for k in (
                "server_turn_detection",
                "server_barge_in",
                "emits_interrupt_event",
                "emits_caller_transcript",
                "emits_agent_transcript",
                "emits_turn_start",
                "input_sample_rate_hz",
                "output_sample_rate_hz",
                "input_mime",
                "output_mime",
                "native_tool_calling",
                "notes",
            )
        }


class ProviderUnavailable(RuntimeError):
    """Raised when a provider cannot start: no credentials, no quota, no wire."""


# --------------------------------------------------------------------------
# Session / provider protocols
# --------------------------------------------------------------------------
class VoiceSession(ABC):
    """One live bidirectional call."""

    capabilities: ProviderCapabilities
    clock: Clock

    @abstractmethod
    async def send_audio(self, pcm: PCM, *, ground_truth_text: str | None = None) -> None:
        """Push caller microphone audio.

        Resampling to the provider's input rate is the adapter's job, not the
        caller simulator's.

        ``ground_truth_text`` is what the caller simulator actually asked the
        TTS to say. Live adapters ignore it on the wire, but the harness records
        it next to the provider's own ASR of the same audio, which is how caller
        word error rate gets measured -- a real Experience signal, and one that
        is free because the harness authored the utterance. Offline simulators
        also use it to produce realistic transcript frames.
        """

    @abstractmethod
    async def commit_turn(self) -> None:
        """Signal end-of-caller-turn.

        A no-op for providers doing server-side turn detection. Present on the
        interface unconditionally so the orchestrator never branches on vendor.
        """

    @abstractmethod
    async def send_tool_result(self, call_id: str, name: str, payload: Any) -> None: ...

    @abstractmethod
    async def cancel_response(self) -> None:
        """Ask the provider to stop speaking now.

        Used to emulate barge-in against providers that do not do it themselves.
        Providers with ``server_barge_in`` ignore this; the difference is
        recorded in the run manifest so the two are never silently compared.
        """

    @abstractmethod
    def events(self) -> AsyncIterator[ServerEvent]: ...

    @abstractmethod
    async def close(self) -> None: ...


class VoiceProvider(ABC):
    """Factory for sessions, plus static facts about the vendor."""

    name: str = "unnamed"
    #: Whether this adapter's wire protocol has been exercised against the live
    #: service. False means the translation layer is unit-tested against
    #: fixtures but nothing has ever round-tripped. Surfaced in the report.
    wire_verified: bool = False

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...

    @abstractmethod
    async def connect(self, config: SessionConfig, clock: Clock | None = None) -> VoiceSession: ...

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wire_verified": self.wire_verified,
            "capabilities": self.capabilities().to_dict(),
        }


_REGISTRY: dict[str, type[VoiceProvider]] = {}


def register_provider(cls: type[VoiceProvider]) -> type[VoiceProvider]:
    _REGISTRY[cls.name] = cls
    return cls


def get_provider(name: str, **kwargs: Any) -> VoiceProvider:
    if name not in _REGISTRY:
        raise KeyError(f"unknown provider {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def registered_providers() -> list[str]:
    return sorted(_REGISTRY)
