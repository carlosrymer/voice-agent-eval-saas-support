"""Barge-in detection from the two audio tracks.

A caller interrupting the agent is the single most load-bearing behaviour in a
voice agent that nobody measures, because it is invisible in a transcript: the
text of a call where the agent talked over the caller for four seconds and the
text of a call where it yielded in 150 ms can be identical.

Detection is deliberately audio-first. A caller speech segment that *begins*
while an agent speech segment is active is a barge-in, full stop -- no provider
cooperation required. Everything after that is graded against the audio too:

* **yielded** -- did the agent's audio actually stop within ``yield_window_ms``
  of the caller starting? This is a choice of threshold, not a fact, and it is a
  parameter with a stated default rather than a magic number in a branch.
* **yield_latency_ms** -- caller onset to the last agent audio sample. This is
  the number that determines whether an interruption feels responsive or rude.
* **overlap_ms** -- how much speech was genuinely simultaneous.

Provider signals are recorded *alongside* the audio measurement, never instead
of it. When a provider emits an interrupt frame, ``signal_latency_ms`` says how
far ahead of (or behind) the actual audio stop that frame was. On more than one
realtime stack the control-plane signal and the audio stream do not agree, and
a harness that trusted the signal would report a yield that the caller never
heard.

State loss is the second half of the question. Yielding fast is worthless if the
agent then forgets what it was doing. Two things get checked, both mechanically:
whether the agent restarted its interrupted sentence from the top (token overlap
against the utterance it abandoned) and whether a tool call that had been
requested before the interruption never received a result.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from voiceval.audio.vad import Segment, VadConfig
from voiceval.metrics.timeline import CallRecord
from voiceval.providers.base import EventKind

#: An agent that stops within this of the caller's onset counts as having
#: yielded. 800 ms is roughly the point at which a human caller stops assuming
#: the line is broken and starts assuming they were ignored.
DEFAULT_YIELD_WINDOW_MS = 800.0

#: Token-overlap above this between the abandoned utterance and the next one
#: means the agent restarted rather than resumed.
RESTART_OVERLAP = 0.6

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str | None) -> list[str]:
    return _WORD.findall((text or "").lower())


def _overlap_ratio(a: str | None, b: str | None) -> float:
    """Fraction of the shorter utterance's tokens present in the other."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


@dataclass
class BargeIn:
    caller_onset_t: float
    #: The agent segment that was in progress when the caller started.
    agent_segment_start_t: float
    agent_stop_t: float
    yield_latency_ms: float
    overlap_ms: float
    yielded: bool
    #: Provider emitted an explicit interrupt frame for this event.
    provider_signalled: bool
    #: Interrupt frame time minus audio stop time. Negative means the control
    #: plane announced the stop before the audio actually stopped.
    signal_latency_ms: float | None
    #: Did the agent resume speaking, and how long after being cut off.
    resumed_after_ms: float | None
    #: Token overlap between the abandoned utterance and the next agent turn.
    restart_overlap: float | None
    restarted: bool
    #: A tool call requested before the barge-in that never got a result.
    dropped_tool_calls: list[str] = field(default_factory=list)

    @property
    def state_preserved(self) -> bool:
        return not self.restarted and not self.dropped_tool_calls

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state_preserved"] = self.state_preserved
        return d


@dataclass
class BargeInReport:
    events: list[BargeIn]
    n_barge_ins: int
    n_yielded: int
    n_state_preserved: int
    yield_latency_ms_p50: float | None
    yield_latency_ms_p95: float | None
    overlap_ms_total: float
    #: Caller utterances the harness deliberately started mid-agent-turn. If
    #: this is greater than n_barge_ins, some scripted interruptions did not
    #: actually land on top of agent speech and the run under-tested barge-in.
    n_scripted_barge_ins: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_barge_ins": self.n_barge_ins,
            "n_scripted_barge_ins": self.n_scripted_barge_ins,
            "n_yielded": self.n_yielded,
            "n_state_preserved": self.n_state_preserved,
            "yield_rate": (self.n_yielded / self.n_barge_ins) if self.n_barge_ins else None,
            "state_preserved_rate": (
                (self.n_state_preserved / self.n_barge_ins) if self.n_barge_ins else None
            ),
            "yield_latency_ms_p50": self.yield_latency_ms_p50,
            "yield_latency_ms_p95": self.yield_latency_ms_p95,
            "overlap_ms_total": self.overlap_ms_total,
            "events": [e.to_dict() for e in self.events],
        }


def _agent_segment_active_at(segments: list[Segment], t: float) -> Segment | None:
    for s in segments:
        if s.start_s < t < s.end_s:
            return s
    return None


def detect_barge_ins(
    record: CallRecord,
    vad_cfg: VadConfig = VadConfig(),
    yield_window_ms: float = DEFAULT_YIELD_WINDOW_MS,
) -> BargeInReport:
    caller_segs = record.caller_vad(vad_cfg).segments
    agent_segs = record.agent_vad(vad_cfg).segments
    interrupt_events = record.events_of(EventKind.INTERRUPTED)

    events: list[BargeIn] = []
    for cs in caller_segs:
        active = _agent_segment_active_at(agent_segs, cs.start_s)
        if active is None:
            continue

        stop_t = active.end_s
        yield_ms = (stop_t - cs.start_s) * 1000.0
        overlap_ms = cs.overlap_s(active) * 1000.0

        near = [e for e in interrupt_events if abs(e.t - stop_t) <= 1.5]
        signalled = bool(near)
        signal_latency = (min(near, key=lambda e: abs(e.t - stop_t)).t - stop_t) * 1000.0 if near else None

        later = [s for s in agent_segs if s.start_s > stop_t + 1e-9]
        resumed_after = (later[0].start_s - stop_t) * 1000.0 if later else None

        abandoned = _utterance_overlapping(record, active)
        following = _utterance_starting_after(record, stop_t)
        overlap_ratio = (
            _overlap_ratio(abandoned.text if abandoned else None,
                           following.text if following else None)
            if abandoned and following
            else None
        )

        dropped = [
            te.call_id
            for te in record.tool_executions
            if te.requestor == "assistant"
            and te.requested_t < cs.start_s
            and te.finished_t <= te.started_t
        ]

        events.append(
            BargeIn(
                caller_onset_t=cs.start_s,
                agent_segment_start_t=active.start_s,
                agent_stop_t=stop_t,
                yield_latency_ms=yield_ms,
                overlap_ms=overlap_ms,
                yielded=yield_ms <= yield_window_ms,
                provider_signalled=signalled,
                signal_latency_ms=signal_latency,
                resumed_after_ms=resumed_after,
                restart_overlap=overlap_ratio,
                restarted=bool(overlap_ratio is not None and overlap_ratio >= RESTART_OVERLAP),
                dropped_tool_calls=dropped,
            )
        )

    lat = sorted(e.yield_latency_ms for e in events)

    def rank(p: float) -> float | None:
        if not lat:
            return None
        import math

        return lat[min(len(lat) - 1, max(0, math.ceil(p * len(lat)) - 1))]

    return BargeInReport(
        events=events,
        n_barge_ins=len(events),
        n_yielded=sum(1 for e in events if e.yielded),
        n_state_preserved=sum(1 for e in events if e.state_preserved),
        yield_latency_ms_p50=rank(0.5),
        yield_latency_ms_p95=rank(0.95),
        overlap_ms_total=sum(e.overlap_ms for e in events),
        n_scripted_barge_ins=sum(1 for u in record.caller_utterances if u.is_barge_in),
    )


def _utterance_overlapping(record: CallRecord, seg: Segment):
    """The agent utterance this detected speech segment belongs to.

    Matched by greatest temporal overlap rather than by containing a point.
    VAD boundaries land a frame or two either side of the bookkeeping times,
    so a point test at the segment edge misses the utterance it is obviously
    part of -- which silently disabled restart detection.
    """
    best, best_overlap = None, 0.0
    for a in record.agent_utterances:
        if a.audio_start_t is None or a.audio_end_t is None:
            continue
        ov = seg.overlap_s(Segment(a.audio_start_t, a.audio_end_t))
        if ov > best_overlap:
            best, best_overlap = a, ov
    return best


def _utterance_starting_after(record: CallRecord, t: float):
    later = [
        a
        for a in record.agent_utterances
        if a.audio_start_t is not None and a.audio_start_t > t + 1e-9
    ]
    return min(later, key=lambda a: a.audio_start_t) if later else None
