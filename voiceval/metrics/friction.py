"""Conversational friction, measured rather than judged.

The article this project implements lists friction indicators -- repeated
requests to repeat, clarification loops, long silences, overlapping speech, call
duration anomalies, early termination -- and suggests an LLM judge for the
qualitative end. Almost all of them turn out to be computable, and this module
computes them. What is left over for a judge afterwards is genuinely subjective
(does the pacing feel natural), which is a much smaller and much more honestly
labelled claim than "the judge scored experience 4.2/5".

Two of these deserve their caveats stated where the code lives.

**Clarification and repeat requests are phrase-matched.** A regex list is a
recall-limited instrument: it catches "sorry, could you repeat that" and misses
a novel paraphrase. It is used here anyway because the alternative -- asking a
model -- makes the number depend on the same model family being evaluated, and
because a false negative is at least a *stable* false negative that does not
drift between runs. The patterns are data, they are exported in the report, and
:func:`unmatched_question_rate` reports how much agent speech was interrogative
but unmatched, which is the honest bound on what the list is missing.

**Word error rate is free here and is not usually free.** The harness authored
every caller utterance before sending it to TTS, so the ground truth is known
exactly and the provider's ASR can be scored against it without annotation.
That makes it one of the few Experience numbers in this project with an
uncontestable reference.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from voiceval.audio.vad import Segment, VadConfig, gaps_between, overlap_segments
from voiceval.metrics.timeline import CallRecord

#: Silence longer than this, with neither party speaking, is dead air.
DEFAULT_SILENCE_THRESHOLD_S = 1.0

#: Agent utterance pairs above this Jaccard similarity count as a repetition.
REPETITION_JACCARD = 0.75

REPEAT_REQUEST_PATTERNS = [
    r"\b(can|could) you (please )?(repeat|say) (that|it) (again|one more time)\b",
    r"\bsay (that|it) again\b",
    r"\b(i )?(didn'?t|did not) (quite )?(catch|hear|get) (that|you)\b",
    r"\bcome again\b",
    r"\bone more time\b",
    r"\bsorry,? what\b",
    r"\bpardon( me)?\b",
    r"\byou'?re breaking up\b",
]

CLARIFICATION_PATTERNS = [
    r"\bjust to (confirm|clarify|make sure)\b",
    r"\bdid you (say|mean)\b",
    r"\bcan you clarify\b",
    r"\bwhat do you mean by\b",
    r"\bso you'?re saying\b",
    r"\bis that (right|correct)\b",
    r"\blet me make sure i (understand|have that right)\b",
]

_REPEAT_RE = [re.compile(p, re.I) for p in REPEAT_REQUEST_PATTERNS]
_CLARIFY_RE = [re.compile(p, re.I) for p in CLARIFICATION_PATTERNS]
_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str | None) -> list[str]:
    return _WORD.findall((text or "").lower())


def _matches(text: str | None, patterns: list[re.Pattern]) -> list[str]:
    t = text or ""
    return [p.pattern for p in patterns if p.search(t)]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, normalized by reference length."""
    r, h = _tokens(reference), _tokens(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, start=1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[len(h)] / len(r)


@dataclass
class FrictionReport:
    # -- dead air --
    n_long_silences: int
    long_silence_threshold_s: float
    longest_silence_s: float | None
    total_silence_s: float
    silence_share: float | None
    silences: list[dict[str, float]] = field(default_factory=list)

    # -- simultaneous speech --
    overlap_total_s: float = 0.0
    overlap_share: float | None = None
    n_overlaps: int = 0

    # -- explicit friction phrases --
    agent_repeat_requests: int = 0
    caller_repeat_requests: int = 0
    agent_clarifications: int = 0
    caller_clarifications: int = 0
    matched_phrases: list[dict[str, Any]] = field(default_factory=list)
    #: Agent questions that matched no pattern, as a bound on recall.
    unmatched_agent_questions: int = 0

    # -- repetition --
    n_agent_repetitions: int = 0
    agent_repetition_pairs: list[dict[str, Any]] = field(default_factory=list)

    # -- cadence --
    n_caller_turns: int = 0
    n_agent_turns: int = 0
    caller_speech_s: float = 0.0
    agent_speech_s: float = 0.0
    speech_ratio_agent_to_caller: float | None = None
    mean_agent_turn_s: float | None = None
    mean_caller_turn_s: float | None = None
    call_duration_s: float = 0.0

    # -- termination --
    ended_reason: str = "completed"
    early_termination: bool = False

    # -- ASR quality --
    caller_wer: float | None = None
    caller_wer_n_utterances: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def friction_report(
    record: CallRecord,
    vad_cfg: VadConfig = VadConfig(),
    silence_threshold_s: float = DEFAULT_SILENCE_THRESHOLD_S,
) -> FrictionReport:
    caller_segs = record.caller_vad(vad_cfg).segments
    agent_segs = record.agent_vad(vad_cfg).segments

    # Dead air is measured on the union of both tracks: a gap only counts when
    # *nobody* is talking. A pause while the agent is mid-sentence is not dead
    # air, it is a sentence.
    merged = _merge(sorted(caller_segs + agent_segs))
    gaps = gaps_between(merged)
    long_gaps = [g for g in gaps if g.duration_s >= silence_threshold_s]

    conv_start = merged[0].start_s if merged else 0.0
    conv_end = merged[-1].end_s if merged else 0.0
    conv_span = max(0.0, conv_end - conv_start)

    overlaps = overlap_segments(caller_segs, agent_segs)
    overlap_total = sum(o.duration_s for o in overlaps)

    caller_speech = sum(s.duration_s for s in caller_segs)
    agent_speech = sum(s.duration_s for s in agent_segs)

    rep = FrictionReport(
        n_long_silences=len(long_gaps),
        long_silence_threshold_s=silence_threshold_s,
        longest_silence_s=max((g.duration_s for g in gaps), default=None),
        total_silence_s=sum(g.duration_s for g in gaps),
        silence_share=(sum(g.duration_s for g in gaps) / conv_span) if conv_span > 0 else None,
        silences=[
            {"start_s": round(g.start_s, 3), "duration_s": round(g.duration_s, 3)}
            for g in long_gaps
        ],
        overlap_total_s=overlap_total,
        overlap_share=(overlap_total / conv_span) if conv_span > 0 else None,
        n_overlaps=len(overlaps),
        n_caller_turns=len(caller_segs),
        n_agent_turns=len(agent_segs),
        caller_speech_s=caller_speech,
        agent_speech_s=agent_speech,
        speech_ratio_agent_to_caller=(agent_speech / caller_speech) if caller_speech > 0 else None,
        mean_agent_turn_s=(agent_speech / len(agent_segs)) if agent_segs else None,
        mean_caller_turn_s=(caller_speech / len(caller_segs)) if caller_segs else None,
        call_duration_s=record.duration_s or conv_end,
        ended_reason=record.ended_reason,
        early_termination=record.ended_reason
        in {"caller_hung_up", "max_turns", "error", "provider_error"},
    )

    for a in record.agent_utterances:
        for pat in _matches(a.text, _REPEAT_RE):
            rep.agent_repeat_requests += 1
            rep.matched_phrases.append({"role": "agent", "kind": "repeat", "pattern": pat,
                                        "text": a.text})
        for pat in _matches(a.text, _CLARIFY_RE):
            rep.agent_clarifications += 1
            rep.matched_phrases.append({"role": "agent", "kind": "clarify", "pattern": pat,
                                        "text": a.text})
        if "?" in (a.text or "") and not _matches(a.text, _REPEAT_RE + _CLARIFY_RE):
            rep.unmatched_agent_questions += 1

    for u in record.caller_utterances:
        for pat in _matches(u.text, _REPEAT_RE):
            rep.caller_repeat_requests += 1
            rep.matched_phrases.append({"role": "caller", "kind": "repeat", "pattern": pat,
                                        "text": u.text})
        for pat in _matches(u.text, _CLARIFY_RE):
            rep.caller_clarifications += 1
            rep.matched_phrases.append({"role": "caller", "kind": "clarify", "pattern": pat,
                                        "text": u.text})

    utts = [a for a in record.agent_utterances if _tokens(a.text)]
    for i in range(len(utts)):
        for j in range(i + 1, len(utts)):
            si, sj = set(_tokens(utts[i].text)), set(_tokens(utts[j].text))
            jac = len(si & sj) / len(si | sj) if (si | sj) else 0.0
            if jac >= REPETITION_JACCARD:
                rep.n_agent_repetitions += 1
                rep.agent_repetition_pairs.append(
                    {"a": utts[i].index, "b": utts[j].index, "jaccard": round(jac, 4)}
                )

    scored = [u for u in record.caller_utterances if u.asr_text is not None and u.text]
    if scored:
        rep.caller_wer = sum(word_error_rate(u.text, u.asr_text or "") for u in scored) / len(scored)
        rep.caller_wer_n_utterances = len(scored)

    return rep


def _merge(segments: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for s in segments:
        if out and s.start_s <= out[-1].end_s:
            out[-1] = Segment(out[-1].start_s, max(out[-1].end_s, s.end_s))
        else:
            out.append(s)
    return out


def unmatched_question_rate(report: FrictionReport) -> float | None:
    """Share of agent questions the phrase lists did not classify."""
    matched = report.agent_repeat_requests + report.agent_clarifications
    total = matched + report.unmatched_agent_questions
    return (report.unmatched_agent_questions / total) if total else None
