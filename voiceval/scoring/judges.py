"""The judge-modality experiment: same model, same rubric, different senses.

The article claims audio-aware judges can assess properties transcript-only
judges "cannot reliably evaluate". This module is built to test that claim
rather than assume it, and the design is chosen to isolate one variable.

**Same model on both sides.** Both judges are the same Gemini model. If one arm
used a text model and the other an audio model, a difference could be the model.
Here the model, the rubric, the task context and the tool ledger are all
identical; the *only* difference is whether the conversation arrives as text or
as a recording. A separate control run uses a second audio model, which tests
judge identity the way a sibling text project did -- where swapping the judge
three ways moved the measured result not at all.

**Abstention is a first-class output.** Three of the six criteria --
pronunciation, pacing, naturalness -- are physically unobservable in a
transcript. Both judges are told, in the same words, to return ``null`` for
anything they cannot assess from what they were given. So the interesting result
is not only "do the scores differ" but "does the transcript judge *know* it
cannot hear". A transcript judge that confidently rates pronunciation 4/5 is
confabulating, and that is measurable rather than arguable.

**Two rubrics.** Narrow (explicit behavioural anchors) and broad (holistic).
The same sibling project found that rubric choice moved a headline number from
0% to 100% while judge identity moved it not at all. Running both here tests
whether modality behaves more like the rubric or more like the judge.

Everything a judge produces is labelled single-family: the agent under test, the
caller's voice, and both judges are all Google models. That is a real
credibility limit on every subjective score in this project, and it is why the
Experience axis is built mostly on measurement and only topped off with these.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from voiceval.audio.pcm import mix, resample, to_wav_bytes
from voiceval.llm import (
    GeminiClient,
    audio_part,
    parse_json_object,
    salvage_scores,
    text_part,
    user_content,
)
from voiceval.metrics.timeline import CallRecord

#: Judge audio is downmixed to one 16 kHz mono track. Mono-mixing is deliberate:
#: it keeps overlapping speech audible as overlap, which is precisely the
#: property a transcript cannot represent and the thing being tested.
JUDGE_AUDIO_RATE = 16000

DEFAULT_JUDGE_MODEL = "gemini-3.6-flash"
DEFAULT_CONTROL_AUDIO_MODEL = "gemini-omni-flash-preview"


@dataclass(frozen=True)
class Criterion:
    id: str
    question: str
    #: True when the property lives in the recording and cannot honestly be
    #: read off a transcript.
    audio_only: bool
    anchors: str = ""


NARROW_CRITERIA = (
    Criterion(
        "pronunciation",
        "Was the agent's speech clearly pronounced and easy to understand?",
        True,
        "5 = every word intelligible including account IDs and amounts. "
        "3 = mostly clear, one or two words needed effort. "
        "1 = frequently unintelligible.",
    ),
    Criterion(
        "pacing",
        "Did the agent's speaking pace and pausing feel natural?",
        True,
        "5 = natural pace with pauses in sensible places. "
        "3 = noticeably fast, slow or evenly monotone. "
        "1 = so rushed or so halting it impeded understanding.",
    ),
    Criterion(
        "naturalness",
        "Did the agent sound like a person speaking rather than text being read out?",
        True,
        "5 = conversational. 3 = clearly synthetic but acceptable. "
        "1 = reads out formatting, lists or punctuation.",
    ),
    Criterion(
        "interruption_handling",
        "When the caller interrupted, did the agent stop promptly and pick up "
        "where the caller took it?",
        False,
        "5 = stopped immediately and responded to the interruption. "
        "3 = stopped late or restarted its previous sentence. "
        "1 = talked over the caller. "
        "Return null if the caller never interrupted.",
    ),
    Criterion(
        "turn_taking",
        "Was turn-taking clean, without awkward silences or people talking over "
        "each other?",
        False,
        "5 = clean hand-offs throughout. 3 = one or two awkward gaps or collisions. "
        "1 = persistent collisions or dead air.",
    ),
    Criterion(
        "overall_experience",
        "Overall, would a customer be satisfied with how this call went as a "
        "conversation, setting aside whether the request was granted?",
        False,
        "5 = smooth and easy. 3 = usable but irritating. 1 = a bad call.",
    ),
)

BROAD_CRITERIA = tuple(
    Criterion(c.id, c.question, c.audio_only, "") for c in NARROW_CRITERIA
)


@dataclass(frozen=True)
class Rubric:
    name: str
    criteria: tuple[Criterion, ...]
    preamble: str

    def prompt(self) -> str:
        lines = [self.preamble, "", "## Criteria", ""]
        for c in self.criteria:
            lines.append(f"### {c.id}")
            lines.append(c.question)
            if c.anchors:
                lines.append(f"Scale: {c.anchors}")
            lines.append("")
        lines += [
            "## Output",
            "",
            "Reply with JSON only, no prose outside it:",
            "",
            "{",
            '  "scores": {',
            '    "<criterion_id>": {"score": <1-5 or null>, "rationale": "<one sentence>"}',
            "  }",
            "}",
            "",
            "Score each criterion from 1 to 5.",
            "",
            "**If you cannot assess a criterion from the material you were given, "
            'set "score" to null and say why in the rationale.* Do not guess, and '
            "do not infer a value you have no evidence for. Returning null is the "
            "correct answer when the evidence is absent.",
        ]
        return "\n".join(lines)


NARROW = Rubric(
    "narrow",
    NARROW_CRITERIA,
    "You are evaluating the *experience* of a recorded customer-support phone "
    "call: how the conversation felt, not whether the company's policy decision "
    "was right. Judge only what the material you are given lets you observe.",
)

BROAD = Rubric(
    "broad",
    BROAD_CRITERIA,
    "You are evaluating the quality of a customer-support phone call. Rate the "
    "following aspects of the caller's experience.",
)

RUBRICS = {"narrow": NARROW, "broad": BROAD}


@dataclass
class CriterionScore:
    score: float | None
    rationale: str = ""

    @property
    def abstained(self) -> bool:
        return self.score is None


@dataclass
class JudgeResult:
    call_id: str
    task_id: str
    #: "transcript" or "audio".
    modality: str
    rubric: str
    model: str
    scores: dict[str, CriterionScore] = field(default_factory=dict)
    error: str | None = None
    raw_text: str = ""
    #: True when strict JSON parsing failed and scores were recovered by regex.
    salvaged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "task_id": self.task_id,
            "modality": self.modality,
            "rubric": self.rubric,
            "model": self.model,
            "error": self.error,
            "salvaged": self.salvaged,
            "scores": {
                k: {"score": v.score, "rationale": v.rationale, "abstained": v.abstained}
                for k, v in self.scores.items()
            },
        }


def shared_context(record: CallRecord) -> str:
    """Non-audio context given identically to both judges.

    Holding this constant is what makes the comparison about modality. Both
    judges know what the call was for and what the agent did; only the
    representation of the conversation itself differs.
    """
    tools = [
        f"  {t.requested_t:7.2f}s  {t.requestor:9} {t.name}("
        f"{json.dumps(t.args, default=str)})"
        for t in sorted(record.tool_executions, key=lambda x: x.requested_t)
    ]
    return "\n".join(
        [
            f"Task: {record.task_id}",
            f"Call duration: {record.duration_s:.1f}s",
            f"Ended because: {record.ended_reason}",
            "",
            "Back-office actions taken during the call (not audible on the line):",
            *(tools or ["  (none)"]),
        ]
    )


def transcript_block(record: CallRecord) -> str:
    lines = []
    for row in record.transcript():
        if row["role"] == "tool":
            continue
        tag = "CALLER" if row["role"] == "caller" else "AGENT "
        extra = " [interrupts]" if row.get("barge_in") else ""
        extra += " [cut off]" if row.get("interrupted") else ""
        lines.append(f"[{row['t']:7.2f}s] {tag}{extra}: {row['text']}")
    return "\n".join(lines) or "(no speech transcribed)"


def call_audio_wav(record: CallRecord) -> bytes:
    caller = record.caller_track
    agent = record.agent_track
    if caller is None and agent is None:
        raise ValueError("call has no audio")
    if caller is None:
        merged = resample(agent, JUDGE_AUDIO_RATE)
    elif agent is None:
        merged = resample(caller, JUDGE_AUDIO_RATE)
    else:
        merged = mix(caller, agent, JUDGE_AUDIO_RATE)
    return to_wav_bytes(merged)


def judge_call(
    record: CallRecord,
    modality: str,
    rubric: Rubric,
    client: GeminiClient,
    model: str = DEFAULT_JUDGE_MODEL,
) -> JudgeResult:
    result = JudgeResult(
        call_id=record.call_id,
        task_id=record.task_id,
        modality=modality,
        rubric=rubric.name,
        model=model,
    )
    parts: list[dict[str, Any]] = [text_part(rubric.prompt()), text_part(shared_context(record))]

    if modality == "transcript":
        parts.append(
            text_part(
                "Here is the full transcript of the call. You do not have the "
                "audio.\n\n" + transcript_block(record)
            )
        )
    elif modality == "audio":
        parts.append(
            text_part(
                "Here is the recording of the call. Both speakers are mixed into "
                "one track, so overlapping speech is audible as overlap. You do "
                "not have a transcript."
            )
        )
        parts.append(audio_part(call_audio_wav(record)))
    else:
        raise ValueError(f"unknown modality {modality!r}")

    try:
        resp = client.generate(
            model,
            [user_content(parts)],
            temperature=0.0,
            max_output_tokens=2000,
            response_mime_type="application/json",
        )
        result.raw_text = resp.text
        try:
            data = parse_json_object(resp.text)
            raw_scores = data.get("scores") or data
        except Exception:
            raw_scores = salvage_scores(resp.text)
            if not raw_scores:
                raise
            result.salvaged = True
        for c in rubric.criteria:
            entry = raw_scores.get(c.id)
            if isinstance(entry, dict):
                score = entry.get("score")
                rationale = str(entry.get("rationale") or "")
            else:
                score, rationale = entry, ""
            result.scores[c.id] = CriterionScore(
                score=None if score is None else float(score), rationale=rationale
            )
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        # A judge failure must never read as a low score; it reads as absent.
        for c in rubric.criteria:
            result.scores[c.id] = CriterionScore(None, "judge error")
    return result


# --------------------------------------------------------------------------
# Disagreement analysis
# --------------------------------------------------------------------------
@dataclass
class CriterionAgreement:
    criterion: str
    audio_only_property: bool
    n_pairs: int
    #: Pairs where both judges returned a number.
    n_both_scored: int
    transcript_abstentions: int
    audio_abstentions: int
    mean_transcript: float | None
    mean_audio: float | None
    mean_abs_delta: float | None
    mean_signed_delta: float | None
    n_disagree_by_1_or_more: int
    max_abs_delta: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_modalities(
    transcript_results: list[JudgeResult], audio_results: list[JudgeResult]
) -> dict[str, Any]:
    """Pair judgements by call and quantify where the two modalities diverge."""
    by_call_t = {r.call_id: r for r in transcript_results}
    by_call_a = {r.call_id: r for r in audio_results}
    shared = sorted(set(by_call_t) & set(by_call_a))

    criteria = {c.id: c for c in NARROW_CRITERIA}
    rows: list[CriterionAgreement] = []
    for cid, crit in criteria.items():
        t_scores, a_scores, deltas = [], [], []
        t_abs = a_abs = 0
        for call_id in shared:
            ts = by_call_t[call_id].scores.get(cid)
            as_ = by_call_a[call_id].scores.get(cid)
            if ts is None or as_ is None:
                continue
            if ts.score is None:
                t_abs += 1
            else:
                t_scores.append(ts.score)
            if as_.score is None:
                a_abs += 1
            else:
                a_scores.append(as_.score)
            if ts.score is not None and as_.score is not None:
                deltas.append(as_.score - ts.score)
        rows.append(
            CriterionAgreement(
                criterion=cid,
                audio_only_property=crit.audio_only,
                n_pairs=len(shared),
                n_both_scored=len(deltas),
                transcript_abstentions=t_abs,
                audio_abstentions=a_abs,
                mean_transcript=(sum(t_scores) / len(t_scores)) if t_scores else None,
                mean_audio=(sum(a_scores) / len(a_scores)) if a_scores else None,
                mean_abs_delta=(sum(abs(d) for d in deltas) / len(deltas)) if deltas else None,
                mean_signed_delta=(sum(deltas) / len(deltas)) if deltas else None,
                n_disagree_by_1_or_more=sum(1 for d in deltas if abs(d) >= 1.0),
                max_abs_delta=max((abs(d) for d in deltas), default=None),
            )
        )

    audio_only = [r for r in rows if r.audio_only_property]
    shared_props = [r for r in rows if not r.audio_only_property]

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    return {
        "n_calls": len(shared),
        "per_criterion": [r.to_dict() for r in rows],
        "summary": {
            "audio_only_properties": {
                "criteria": [r.criterion for r in audio_only],
                "transcript_abstention_rate": (
                    sum(r.transcript_abstentions for r in audio_only)
                    / max(1, sum(r.n_pairs for r in audio_only))
                ),
                "audio_abstention_rate": (
                    sum(r.audio_abstentions for r in audio_only)
                    / max(1, sum(r.n_pairs for r in audio_only))
                ),
                "mean_abs_delta": _mean([r.mean_abs_delta for r in audio_only]),
            },
            "shared_properties": {
                "criteria": [r.criterion for r in shared_props],
                "transcript_abstention_rate": (
                    sum(r.transcript_abstentions for r in shared_props)
                    / max(1, sum(r.n_pairs for r in shared_props))
                ),
                "audio_abstention_rate": (
                    sum(r.audio_abstentions for r in shared_props)
                    / max(1, sum(r.n_pairs for r in shared_props))
                ),
                "mean_abs_delta": _mean([r.mean_abs_delta for r in shared_props]),
            },
        },
    }


def compare_rubrics(narrow: list[JudgeResult], broad: list[JudgeResult]) -> dict[str, Any]:
    """How much does the rubric move the score, holding modality fixed?"""
    by_n = {r.call_id: r for r in narrow}
    by_b = {r.call_id: r for r in broad}
    shared = sorted(set(by_n) & set(by_b))
    out = {}
    for c in NARROW_CRITERIA:
        deltas = []
        for call_id in shared:
            a = by_n[call_id].scores.get(c.id)
            b = by_b[call_id].scores.get(c.id)
            if a and b and a.score is not None and b.score is not None:
                deltas.append(b.score - a.score)
        out[c.id] = {
            "n": len(deltas),
            "mean_abs_delta": (sum(abs(d) for d in deltas) / len(deltas)) if deltas else None,
            "mean_signed_delta": (sum(deltas) / len(deltas)) if deltas else None,
        }
    vals = [v["mean_abs_delta"] for v in out.values() if v["mean_abs_delta"] is not None]
    return {
        "n_calls": len(shared),
        "per_criterion": out,
        "mean_abs_delta": (sum(vals) / len(vals)) if vals else None,
    }
