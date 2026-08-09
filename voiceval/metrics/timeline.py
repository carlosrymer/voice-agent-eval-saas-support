"""The single record every metric, judge and report reads.

A :class:`CallRecord` is what one voice call leaves behind: two audio tracks on
a common session clock, the normalized provider event stream, the harness's own
exact record of every tool it executed, and the caller utterances the harness
authored. Nothing downstream talks to a provider; everything talks to this.

The separation that matters is between *harness-owned* and *provider-reported*
facts. Tool execution timings are harness-owned -- I call the function, so I
know to the microsecond when it started and finished, and no vendor can be
wrong about it. Turn boundaries and transcripts are provider-reported and may be
absent. Audio is neither: it is measured off the recording by a VAD, which makes
it the one timing signal available identically for every provider, including
ones that report nothing. Where those three disagree, the metrics prefer audio
for anything that describes what the caller perceived, and harness-owned data
for anything that describes what the system did.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from voiceval.audio.pcm import PCM, read_wav, write_wav
from voiceval.audio.vad import Segment, VadConfig, VadResult, detect
from voiceval.providers.base import EventKind, ProviderCapabilities, ServerEvent


@dataclass
class ToolExecution:
    """One tool call, timed by the harness that ran it."""

    call_id: str
    name: str
    args: dict[str, Any]
    requestor: str = "assistant"
    #: When the provider asked for it (provider event time).
    requested_t: float = 0.0
    #: When the harness began and finished executing it (harness clock).
    started_t: float = 0.0
    finished_t: float = 0.0
    ok: bool = True
    result: str = ""
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.finished_t - self.started_t) * 1000.0)


@dataclass
class CallerUtterance:
    """One thing the simulated caller said, and when it was on the wire."""

    index: int
    text: str
    start_t: float
    end_t: float
    #: True when this utterance was deliberately started while the agent was
    #: still speaking, i.e. a scripted barge-in.
    is_barge_in: bool = False
    #: The provider's ASR of this utterance, if it reported one.
    asr_text: str | None = None
    asr_final_t: float | None = None


@dataclass
class AgentUtterance:
    index: int
    text: str
    #: First audio byte of this turn and last audio byte actually emitted.
    audio_start_t: float | None = None
    audio_end_t: float | None = None
    turn_started_t: float | None = None
    completed_t: float | None = None
    interrupted: bool = False


@dataclass
class CallRecord:
    call_id: str
    task_id: str
    trial: int
    provider: str
    model: str
    #: Whether this call came off a real wire or an offline simulator. Every
    #: report groups on this, so synthetic runs can never be shown as real ones.
    synthetic: bool
    capabilities: dict[str, Any] = field(default_factory=dict)
    events: list[ServerEvent] = field(default_factory=list)
    tool_executions: list[ToolExecution] = field(default_factory=list)
    caller_utterances: list[CallerUtterance] = field(default_factory=list)
    agent_utterances: list[AgentUtterance] = field(default_factory=list)
    caller_track: PCM | None = None
    agent_track: PCM | None = None
    duration_s: float = 0.0
    ended_reason: str = "completed"
    errors: list[str] = field(default_factory=list)
    #: Anything the run wants to remember: seeds, config, cost counters.
    meta: dict[str, Any] = field(default_factory=dict)

    # -- derived views -----------------------------------------------------
    def caller_vad(self, cfg: VadConfig = VadConfig()) -> VadResult:
        return detect(self.caller_track or PCM(b"", 16000), cfg)

    def agent_vad(self, cfg: VadConfig = VadConfig()) -> VadResult:
        return detect(self.agent_track or PCM(b"", 24000), cfg)

    def events_of(self, *kinds: EventKind) -> list[ServerEvent]:
        ks = set(kinds)
        return [e for e in self.events if e.kind in ks]

    def transcript(self) -> list[dict[str, Any]]:
        """Interleaved caller/agent transcript in session-time order."""
        rows: list[dict[str, Any]] = []
        for u in self.caller_utterances:
            rows.append(
                {
                    "role": "caller",
                    "t": u.start_t,
                    "end_t": u.end_t,
                    "text": u.text,
                    "asr_text": u.asr_text,
                    "barge_in": u.is_barge_in,
                }
            )
        for a in self.agent_utterances:
            rows.append(
                {
                    "role": "agent",
                    "t": a.audio_start_t if a.audio_start_t is not None else (a.turn_started_t or 0.0),
                    "end_t": a.audio_end_t,
                    "text": a.text,
                    "interrupted": a.interrupted,
                }
            )
        for te in self.tool_executions:
            rows.append(
                {
                    "role": "tool",
                    "t": te.requested_t,
                    "end_t": te.finished_t,
                    "text": f"{te.name}({json.dumps(te.args, default=str)})",
                    "result": te.result,
                    "ok": te.ok,
                    "requestor": te.requestor,
                }
            )
        return sorted(rows, key=lambda r: r["t"])

    # -- persistence -------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self.caller_track is not None:
            write_wav(str(d / f"{self.call_id}.caller.wav"), self.caller_track)
        if self.agent_track is not None:
            write_wav(str(d / f"{self.call_id}.agent.wav"), self.agent_track)
        payload = {
            "call_id": self.call_id,
            "task_id": self.task_id,
            "trial": self.trial,
            "provider": self.provider,
            "model": self.model,
            "synthetic": self.synthetic,
            "capabilities": self.capabilities,
            "duration_s": self.duration_s,
            "ended_reason": self.ended_reason,
            "errors": self.errors,
            "meta": self.meta,
            "events": [e.to_dict() for e in self.events],
            "tool_executions": [asdict(t) for t in self.tool_executions],
            "caller_utterances": [asdict(u) for u in self.caller_utterances],
            "agent_utterances": [asdict(a) for a in self.agent_utterances],
            "audio": {
                "caller_wav": f"{self.call_id}.caller.wav" if self.caller_track else None,
                "agent_wav": f"{self.call_id}.agent.wav" if self.agent_track else None,
            },
        }
        p = d / f"{self.call_id}.json"
        p.write_text(json.dumps(payload, indent=2, default=str))
        return p

    @classmethod
    def load(cls, json_path: str | Path) -> "CallRecord":
        p = Path(json_path)
        raw = json.loads(p.read_text())
        rec = cls(
            call_id=raw["call_id"],
            task_id=raw["task_id"],
            trial=raw["trial"],
            provider=raw["provider"],
            model=raw["model"],
            synthetic=raw["synthetic"],
            capabilities=raw.get("capabilities", {}),
            duration_s=raw.get("duration_s", 0.0),
            ended_reason=raw.get("ended_reason", "completed"),
            errors=raw.get("errors", []),
            meta=raw.get("meta", {}),
            tool_executions=[ToolExecution(**t) for t in raw.get("tool_executions", [])],
            caller_utterances=[CallerUtterance(**u) for u in raw.get("caller_utterances", [])],
            agent_utterances=[AgentUtterance(**a) for a in raw.get("agent_utterances", [])],
        )
        # Restoring the event stream matters: `--score-only` re-scores calls
        # loaded from disk, and several latency stages are defined by provider
        # frames. Dropping them here made a reloaded call score differently from
        # the same call in memory -- so the committed artifacts would not have
        # reproduced the published numbers.
        rec.events = [_event_from_dict(e) for e in raw.get("events", [])]
        audio = raw.get("audio") or {}
        if audio.get("caller_wav") and (p.parent / audio["caller_wav"]).exists():
            rec.caller_track = read_wav(str(p.parent / audio["caller_wav"]))
        if audio.get("agent_wav") and (p.parent / audio["agent_wav"]).exists():
            rec.agent_track = read_wav(str(p.parent / audio["agent_wav"]))
        return rec


def _event_from_dict(d: dict[str, Any]) -> ServerEvent:
    """Rebuild an event from its stored form.

    Audio payloads are not stored in the JSON -- the WAV tracks hold them -- so
    a restored audio event carries an empty buffer at the right sample rate.
    Nothing downstream reads bytes off an event; the metrics read the tracks.
    """
    rate = d.get("audio_rate_hz")
    return ServerEvent(
        kind=EventKind(d["kind"]),
        t=float(d.get("t") or 0.0),
        seq=int(d.get("seq") or 0),
        audio=PCM(b"", int(rate)) if rate else None,
        text=d.get("text"),
        is_final=bool(d.get("is_final")),
        call_id=d.get("call_id"),
        tool_name=d.get("tool_name"),
        tool_args=d.get("tool_args") or {},
        message=d.get("message"),
    )


def caps_from(capabilities: ProviderCapabilities) -> dict[str, Any]:
    return capabilities.to_dict()


def segments_within(segments: list[Segment], lo: float, hi: float) -> list[Segment]:
    return [s for s in segments if s.start_s < hi and s.end_s > lo]
