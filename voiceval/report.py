"""Turn artifacts into the site's data files. No number is typed by hand.

Also decides which calls ship with audio. Full-fidelity recordings are ~4 MB a
side and there is no encoder in this environment, so shipping every track would
put a few hundred megabytes of WAV in a git repo. Instead a small, explicitly
chosen set of calls is downmixed to 8 kHz mono -- telephone bandwidth, which is
the right fidelity for a drill-down anyway -- and the selection rule is stated
in the output rather than being "the first few".
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from voiceval.audio.pcm import PCM, mix, resample, write_wav
from voiceval.metrics.timeline import CallRecord

ARTIFACTS = Path("artifacts")
CALLS_DIR = ARTIFACTS / "calls"
SITE_DATA = Path("site/data")
AUDIO_DIR = SITE_DATA / "audio"

WEB_AUDIO_RATE = 8000
MAX_AUDIO_CALLS = 6

#: Committed text-channel baseline for the identical 16 tasks, policy and
#: evaluator, from my earlier tau2-bench project. Voice results are compared
#: against these; they are not recomputed here.
TEXT_BASELINE = {
    "source": "tau2-bench-saas-support-policy (public repo, committed results)",
    "domain": "saas_support, 16 tasks, 7-rule policy, identical evaluator",
    "arms": [
        {
            "agent": "gemini-3.6-flash",
            "channel": "text",
            "pass_hat_1": 1.0,
            "pass_hat_4": 1.0,
            "policy_violations": 0,
            "trials": 4,
        },
        {
            "agent": "kimi-k2.7-code",
            "channel": "text",
            "pass_hat_1": 0.938,
            "pass_hat_4": 0.750,
            "policy_violations": 0,
            "trials": 4,
        },
    ],
}


def web_audio(record: CallRecord) -> PCM | None:
    if record.caller_track is None and record.agent_track is None:
        return None
    if record.caller_track is None:
        return resample(record.agent_track, WEB_AUDIO_RATE)
    if record.agent_track is None:
        return resample(record.caller_track, WEB_AUDIO_RATE)
    return mix(record.caller_track, record.agent_track, WEB_AUDIO_RATE)


def pick_audio_calls(rows: list[dict[str, Any]], limit: int = MAX_AUDIO_CALLS) -> list[str]:
    """Choose drill-down calls by what makes them worth listening to.

    Deliberately not "the first N". The set covers a landed barge-in, a policy
    violation, a failed outcome, and the latency extremes, because those are the
    calls where someone reading a number will want to hear whether it is real.
    """
    picked: list[str] = []

    def add(row):
        if row and row["call_id"] not in picked:
            picked.append(row["call_id"])

    def first(pred, key=None, reverse=False):
        c = [r for r in rows if pred(r)]
        if not c:
            return None
        return sorted(c, key=key or (lambda r: 0), reverse=reverse)[0]

    add(first(lambda r: r["barge_in"]["n_barge_ins"] > 0))
    add(first(lambda r: bool(r["execution"]["violation_rules"])))
    add(first(lambda r: not r["outcome"]["passed"]))
    add(first(lambda r: r["latency"]["end_of_turn_ms"]["p50"] is not None,
              key=lambda r: r["latency"]["end_of_turn_ms"]["p50"], reverse=True))
    add(first(lambda r: r["latency"]["end_of_turn_ms"]["p50"] is not None,
              key=lambda r: r["latency"]["end_of_turn_ms"]["p50"]))
    for r in rows:
        if len(picked) >= limit:
            break
        add(r)
    return picked[:limit]


def build(
    results_path: str | Path = ARTIFACTS / "results.json",
    calls_dir: str | Path | None = None,
    site_data: str | Path | None = None,
) -> dict[str, Any]:
    results = json.loads(Path(results_path).read_text())
    calls_dir = Path(calls_dir) if calls_dir else Path(
        results.get("config", {}).get("calls_dir") or CALLS_DIR
    )
    site = Path(site_data) if site_data else SITE_DATA
    audio_out = site / "audio"
    rows = results["calls"]

    site.mkdir(parents=True, exist_ok=True)
    if audio_out.exists():
        shutil.rmtree(audio_out)
    audio_out.mkdir(parents=True, exist_ok=True)

    audio_calls = pick_audio_calls(rows)
    transcripts: dict[str, Any] = {}
    shipped_audio: dict[str, str] = {}

    for row in rows:
        path = calls_dir / f"{row['call_id']}.json"
        if not path.exists():
            continue
        rec = CallRecord.load(path)
        transcripts[rec.call_id] = {
            "call_id": rec.call_id,
            "task_id": rec.task_id,
            "trial": rec.trial,
            "duration_s": rec.duration_s,
            "ended_reason": rec.ended_reason,
            "rows": rec.transcript(),
            "caller_utterances": [
                {
                    "index": u.index,
                    "text": u.text,
                    "asr_text": u.asr_text,
                    "start_t": u.start_t,
                    "end_t": u.end_t,
                    "barge_in": u.is_barge_in,
                }
                for u in rec.caller_utterances
            ],
        }
        if rec.call_id in audio_calls:
            pcm = web_audio(rec)
            if pcm is not None:
                name = f"{rec.call_id}.wav"
                write_wav(str(audio_out / name), pcm)
                shipped_audio[rec.call_id] = f"data/audio/{name}"

    payload = {
        "generated_at": results["generated_at"],
        "config": results["config"],
        "summary": results["summary"],
        "calls": [
            {
                "call_id": r["call_id"],
                "task_id": r["task_id"],
                "trial": r["trial"],
                "duration_s": r["duration_s"],
                "ended_reason": r["ended_reason"],
                "outcome_passed": r["outcome"]["passed"],
                "outcome_reward": r["outcome"]["reward"],
                "action_reward": r["execution"]["action_reward"],
                "violations": r["execution"]["violation_rules"],
                "violation_detail": [
                    {"rule": v["rule"], "detail": v["detail"]}
                    for v in r["execution"]["violations"]
                ],
                "clean": r["execution"]["clean"],
                "eot_p50_ms": r["latency"]["end_of_turn_ms"]["p50"],
                "eot_p95_ms": r["latency"]["end_of_turn_ms"]["p95"],
                "n_turns": r["latency"]["n_turns"],
                "stage_mean_ms": r["latency"].get("stage_mean_ms", {}),
                "barge_in": {
                    k: r["barge_in"][k]
                    for k in ("n_barge_ins", "n_scripted_barge_ins", "n_yielded",
                              "n_state_preserved", "yield_latency_ms_p50")
                },
                "friction": {
                    k: r["friction"][k]
                    for k in (
                        "n_long_silences", "longest_silence_s", "overlap_total_s",
                        "agent_repeat_requests", "caller_repeat_requests",
                        "n_agent_repetitions", "caller_wer", "early_termination",
                        "n_caller_turns", "n_agent_turns",
                    )
                },
                "tools": [
                    {"name": t["name"], "by": t["by"], "t": t["t"]}
                    for t in r["execution"]["ledger"]
                ],
                "audio": shipped_audio.get(r["call_id"]),
            }
            for r in rows
        ],
        "judging": results.get("judging", {}).get("analysis", {}),
        "judge_raw": results.get("judging", {}).get("raw", {}),
        "spend": results.get("spend", {}),
        "text_baseline": TEXT_BASELINE,
        "stage_labels": (rows[0]["latency"].get("stage_labels") if rows else {}),
        "audio_selection_rule": (
            "One call each with a landed barge-in, a policy violation, a failed "
            "outcome, and the slowest and fastest median end-of-turn latency; "
            f"capped at {MAX_AUDIO_CALLS}. Downmixed to {WEB_AUDIO_RATE} Hz mono "
            "so both speakers, and any overlap, are audible in one track."
        ),
    }

    (site / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
    (site / "transcripts.json").write_text(json.dumps(transcripts, indent=2, default=str))
    print(
        f"Wrote {site/'summary.json'} ({len(payload['calls'])} calls) and "
        f"{len(shipped_audio)} audio files"
    )
    return payload


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ARTIFACTS / "results.json"))
    ap.add_argument("--calls-dir", default=None)
    ap.add_argument("--site-data", default=None)
    a = ap.parse_args()
    build(a.results, a.calls_dir, a.site_data)
