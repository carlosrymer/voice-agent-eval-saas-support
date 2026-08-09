"""Latency decomposition, measured off the recording rather than self-reported.

The headline number is **end-of-turn latency**: the gap a caller actually
experiences between finishing their sentence and hearing the agent begin. It is
measured from the two audio tracks, so it is available for every provider,
including one that reports no timing telemetry at all.

Underneath it sits a decomposition into ASR, inference, tool execution and
speech synthesis. That decomposition is only as good as the boundary markers a
provider chooses to emit, and providers differ. The rule this module follows is
that **components always sum to the total**: whatever cannot be attributed to a
named stage is reported as ``unattributed_ms`` rather than being folded into a
neighbouring stage or dropped. A breakdown that silently doesn't add up is worse
than no breakdown, because it invites conclusions about where the time went that
the data does not support.

Percentiles use the nearest-rank method on the sorted sample: P95 of n
observations is the observation at index ``ceil(0.95 * n) - 1``. No
interpolation, so every reported percentile is a latency that actually happened.
With the sample sizes here (tens of turns, not thousands) an interpolated P99 is
a fiction; nearest-rank at least points at a real turn you can go listen to.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from voiceval.audio.vad import Segment, VadConfig
from voiceval.metrics.timeline import CallRecord
from voiceval.providers.base import EventKind

#: A caller pause longer than this is treated as a turn boundary rather than a
#: mid-sentence breath when pairing caller speech with the agent's reply.
TURN_PAIRING_WINDOW_S = 30.0

#: The latency budget, in the order the stages occur. ``unattributed_ms`` is
#: deliberately not in this list: it is the residual that makes the named
#: stages plus itself equal the measured total, and including it here would
#: make it self-referential.
STAGE_NAMES: tuple[str, ...] = (
    "asr_ms",
    "to_turn_start_ms",
    "inference_pre_tool_ms",
    "tool_ms",
    "inter_tool_inference_ms",
    "inference_post_tool_ms",
    "to_first_audio_ms",
)

STAGE_LABELS: dict[str, str] = {
    "asr_ms": "caller speech end -> final ASR",
    "to_turn_start_ms": "-> provider turn start",
    "inference_pre_tool_ms": "-> first tool request",
    "tool_ms": "tool execution (harness-timed)",
    "inter_tool_inference_ms": "model time between tool calls",
    "inference_post_tool_ms": "last tool result -> first audio",
    "to_first_audio_ms": "-> first audio byte",
    "unattributed_ms": "residual, not attributable to a reported marker",
}


@dataclass
class TurnLatency:
    """Latency decomposition for one agent turn."""

    turn_index: int
    caller_speech_end_t: float
    first_audio_t: float | None
    #: The number a caller feels. None when the agent never spoke this turn.
    end_of_turn_ms: float | None
    #: Named stages, in the order they occur. Each stage is named for the pair
    #: of markers that bound it, never for what I assume happened inside it: a
    #: provider that reports no ASR boundary gets ``to_turn_start_ms``, not an
    #: ``inference_ms`` that quietly has speech recognition folded into it.
    #: Any stage is None when a marker defining it is absent.
    asr_ms: float | None = None
    to_turn_start_ms: float | None = None
    inference_pre_tool_ms: float | None = None
    tool_ms: float | None = None
    inter_tool_inference_ms: float | None = None
    inference_post_tool_ms: float | None = None
    to_first_audio_ms: float | None = None
    #: Total minus everything attributed above. Always present when
    #: end_of_turn_ms is present; may be the whole of it.
    unattributed_ms: float | None = None
    #: How long the agent then spoke for. Not part of the latency budget, but
    #: the denominator for "did it get cut off".
    speech_ms: float | None = None
    n_tool_calls: int = 0
    interrupted: bool = False
    #: Why a stage is missing, keyed by stage name.
    missing_reasons: dict[str, str] = field(default_factory=dict)

    def attributed_ms(self) -> float:
        return sum(getattr(self, name) or 0.0 for name in STAGE_NAMES)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Percentiles:
    n: int
    p50: float | None
    p95: float | None
    p99: float | None
    mean: float | None
    min: float | None
    max: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentiles(values: list[float]) -> Percentiles:
    """Nearest-rank percentiles. Every value returned is an observed value."""
    xs = sorted(v for v in values if v is not None and math.isfinite(v))
    if not xs:
        return Percentiles(0, None, None, None, None, None, None)

    def rank(p: float) -> float:
        return xs[min(len(xs) - 1, max(0, math.ceil(p * len(xs)) - 1))]

    return Percentiles(
        n=len(xs),
        p50=rank(0.50),
        p95=rank(0.95),
        p99=rank(0.99),
        mean=sum(xs) / len(xs),
        min=xs[0],
        max=xs[-1],
    )


def _caller_speech_end_before(segments: list[Segment], t: float) -> float | None:
    """End of the last caller speech segment that finished before ``t``."""
    ends = [s.end_s for s in segments if s.end_s <= t + 1e-9]
    return max(ends) if ends else None


def turn_latencies(record: CallRecord, vad_cfg: VadConfig = VadConfig()) -> list[TurnLatency]:
    caller_segs = record.caller_vad(vad_cfg).segments
    caps = record.capabilities or {}
    out: list[TurnLatency] = []

    for turn in record.agent_utterances:
        idx = turn.index
        first_audio = turn.audio_start_t
        anchor = first_audio if first_audio is not None else turn.turn_started_t
        if anchor is None:
            continue

        caller_end = _caller_speech_end_before(caller_segs, anchor)
        if caller_end is None:
            # Agent spoke first (a greeting). There is no caller turn to
            # measure against, so this turn contributes no latency observation.
            continue

        tl = TurnLatency(
            turn_index=idx,
            caller_speech_end_t=caller_end,
            first_audio_t=first_audio,
            end_of_turn_ms=None if first_audio is None else (first_audio - caller_end) * 1000.0,
            speech_ms=(
                (turn.audio_end_t - turn.audio_start_t) * 1000.0
                if turn.audio_end_t is not None and turn.audio_start_t is not None
                else None
            ),
            interrupted=turn.interrupted,
        )

        tools = [
            te
            for te in record.tool_executions
            if te.requestor == "assistant"
            and turn.turn_started_t is not None
            and turn.turn_started_t - 1e-9 <= te.requested_t
            and (first_audio is None or te.requested_t <= first_audio + 1e-9)
        ]
        tl.n_tool_calls = len(tools)

        # Walk the markers in order. `cursor` is the latest boundary that is
        # actually known; each stage spans from it to the next known marker.
        cursor = caller_end

        if caps.get("emits_caller_transcript"):
            finals = [
                e
                for e in record.events_of(EventKind.CALLER_TRANSCRIPT)
                if e.is_final and caller_end - 1e-9 <= e.t <= anchor + 1e-9
            ]
            if finals:
                t_asr = min(f.t for f in finals)
                tl.asr_ms = max(0.0, (t_asr - cursor) * 1000.0)
                cursor = max(cursor, t_asr)
            else:
                tl.missing_reasons["asr_ms"] = "no final caller transcript in this turn window"
        else:
            tl.missing_reasons["asr_ms"] = (
                "provider does not emit caller transcripts; speech recognition "
                "cannot be separated from inference"
            )

        if caps.get("emits_turn_start") and turn.turn_started_t is not None:
            if turn.turn_started_t > cursor:
                tl.to_turn_start_ms = (turn.turn_started_t - cursor) * 1000.0
                cursor = turn.turn_started_t
        else:
            tl.missing_reasons["to_turn_start_ms"] = (
                "provider emits no turn-start frame; response onset is only "
                "observable as the first audio byte"
            )

        if tools:
            first_req = min(t.requested_t for t in tools)
            last_fin = max(t.finished_t for t in tools)
            tl.inference_pre_tool_ms = max(0.0, (first_req - cursor) * 1000.0)
            # `tool_ms` is what the tools themselves cost; the wall-clock span
            # from first request to last result is usually longer because the
            # model thinks between calls. Splitting them keeps the budget
            # additive without charging model time to the tools.
            tl.tool_ms = sum(t.duration_ms for t in tools)
            tool_wall_ms = max(0.0, (last_fin - first_req) * 1000.0)
            tl.inter_tool_inference_ms = max(0.0, tool_wall_ms - tl.tool_ms)
            cursor = max(cursor, last_fin)
            if first_audio is not None:
                tl.inference_post_tool_ms = max(0.0, (first_audio - cursor) * 1000.0)
        elif first_audio is not None:
            tl.to_first_audio_ms = max(0.0, (first_audio - cursor) * 1000.0)

        if tl.end_of_turn_ms is not None:
            tl.unattributed_ms = round(tl.end_of_turn_ms - tl.attributed_ms(), 6)
        out.append(tl)
    return out


@dataclass
class LatencyReport:
    end_of_turn_ms: Percentiles
    stages: dict[str, Percentiles]
    unattributed_ms: Percentiles
    agent_speech_ms: Percentiles
    n_turns: int
    n_turns_with_tools: int
    #: Share of total end-of-turn latency that could not be attributed to a
    #: named stage. The honesty dial: a high value means the decomposition is
    #: mostly a single opaque block, and the report says so.
    unattributed_share: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "end_of_turn_ms": self.end_of_turn_ms.to_dict(),
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "stage_labels": STAGE_LABELS,
            "unattributed_ms": self.unattributed_ms.to_dict(),
            "agent_speech_ms": self.agent_speech_ms.to_dict(),
            "n_turns": self.n_turns,
            "n_turns_with_tools": self.n_turns_with_tools,
            "unattributed_share": self.unattributed_share,
            "stage_mean_ms": self.stage_mean_ms(),
        }

    def stage_mean_ms(self) -> dict[str, float]:
        """Mean contribution of each stage, for a stacked breakdown chart.

        Averaged over *all* turns, not only turns where the stage fired, so the
        bars sum to the mean end-of-turn latency. A stage that only occurs on
        tool turns therefore shows its true share of the whole call, which is
        what somebody deciding where to optimise actually needs.
        """
        n = self.n_turns or 1
        out = {k: (v.mean or 0.0) * v.n / n for k, v in self.stages.items()}
        out["unattributed_ms"] = (self.unattributed_ms.mean or 0.0) * self.unattributed_ms.n / n
        return out


def aggregate(turns: list[TurnLatency]) -> LatencyReport:
    def col(attr: str) -> list[float]:
        return [getattr(t, attr) for t in turns if getattr(t, attr) is not None]

    total = sum(col("end_of_turn_ms"))
    unattr = sum(col("unattributed_ms"))
    return LatencyReport(
        end_of_turn_ms=percentiles(col("end_of_turn_ms")),
        stages={name: percentiles(col(name)) for name in STAGE_NAMES},
        unattributed_ms=percentiles(col("unattributed_ms")),
        agent_speech_ms=percentiles(col("speech_ms")),
        n_turns=len(turns),
        n_turns_with_tools=sum(1 for t in turns if t.n_tool_calls),
        unattributed_share=(unattr / total) if total > 0 else None,
    )


def latency_report(record: CallRecord, vad_cfg: VadConfig = VadConfig()) -> LatencyReport:
    return aggregate(turn_latencies(record, vad_cfg))
