"""The whole experiment, end to end, in one command.

    uv run python -m voiceval.run_experiment --provider gemini_live --trials 1

Runs every task as a voice call, scores all three axes, judges each call under
both modalities and both rubrics, and writes everything needed to rebuild the
report and the site. Nothing downstream re-derives anything from a model: the
report reads the artifacts this writes.

Concurrency is real but modest. Calls are paced in real time, so a task takes as
long as the conversation takes; running a handful at once is the difference
between fifteen minutes and an hour. It is capped because the caller brain, the
agent and both judges all share one API quota, and saturating it turns 429s into
what looks like agent failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import voiceval.providers  # noqa: F401  (registers adapters)
from voiceval.caller.simulator import BargeInPlan
from voiceval.domain import ensure_registered, tasks as load_tasks
from voiceval.llm import GeminiClient
from voiceval.metrics.bargein import detect_barge_ins
from voiceval.metrics.friction import friction_report
from voiceval.metrics.latency import latency_report, turn_latencies
from voiceval.metrics.timeline import CallRecord
from voiceval.orchestrator import CallConfig, run_call
from voiceval.providers.base import get_provider
from voiceval.scoring import judges as J
from voiceval.scoring.execution import score_execution
from voiceval.scoring.outcome import score_outcome
from voiceval.tracing.otel import Tracer
from voiceval.tts import get_tts

ARTIFACTS = Path("artifacts")
CALLS_DIR = ARTIFACTS / "calls"

#: Which agent turns the caller interrupts. Turn 1 is chosen because turn 0 is
#: usually a short greeting with nothing to interrupt, and a plan that never
#: lands would leave the barge-in metrics measured on a sample of zero.
DEFAULT_BARGE_IN_TURNS = (1, 3)


def build_barge_in(spec: str) -> BargeInPlan:
    if spec.strip().lower() in {"", "none", "off"}:
        return BargeInPlan(turns=())
    turns = tuple(int(x) for x in spec.split(",") if x.strip())
    return BargeInPlan(turns=turns, offset_s=1.0)


async def run_one(
    task,
    trial: int,
    provider_name: str,
    provider_kwargs: dict[str, Any],
    tts_name: str,
    client: GeminiClient,
    cfg_kwargs: dict[str, Any],
    sem: asyncio.Semaphore,
    calls_dir: Path = CALLS_DIR,
) -> CallRecord | None:
    async with sem:
        provider = get_provider(provider_name, **provider_kwargs)
        tts = get_tts(tts_name)
        cfg = CallConfig(trial=trial, **cfg_kwargs)
        tracer = Tracer()
        t0 = time.monotonic()
        try:
            with tracer.span(
                "voice.call", kind="CHAIN", task_id=task.id, trial=trial,
                provider=provider_name,
            ):
                record = await run_call(task, provider, tts, client, cfg)
        except Exception as exc:
            print(f"  !! {task.id} trial {trial} crashed: {type(exc).__name__}: {exc}",
                  flush=True)
            return None
        record.meta["wall_clock_s"] = round(time.monotonic() - t0, 2)
        record.meta["tts_calls"] = getattr(tts, "n_calls", 0)
        record.meta["tts_cache_hits"] = getattr(tts, "n_cache_hits", 0)
        record.meta["tts_usage"] = dict(getattr(tts, "usage", {}) or {})
        record.save(calls_dir)
        n_tools = len(record.tool_executions)
        print(
            f"  ok {task.id} t{trial}: {record.ended_reason}, "
            f"{len(record.agent_utterances)} agent turns, {n_tools} tools, "
            f"{record.duration_s:.0f}s",
            flush=True,
        )
        return record


def judge_all(records: list[CallRecord], client: GeminiClient, args) -> dict[str, Any]:
    """Both modalities x both rubrics, plus a second-model audio control."""
    out: dict[str, list[dict[str, Any]]] = {}
    results: dict[str, list[J.JudgeResult]] = {}

    plan = [
        ("transcript_narrow", "transcript", J.NARROW, args.judge_model),
        ("audio_narrow", "audio", J.NARROW, args.judge_model),
    ]
    if not args.no_rubric_sweep:
        plan += [
            ("transcript_broad", "transcript", J.BROAD, args.judge_model),
            ("audio_broad", "audio", J.BROAD, args.judge_model),
        ]
    if args.control_audio_model:
        plan.append(("audio_narrow_control", "audio", J.NARROW, args.control_audio_model))

    for key, modality, rubric, model in plan:
        rs = []
        for rec in records:
            if modality == "audio" and (rec.agent_track is None or rec.caller_track is None):
                continue
            rs.append(J.judge_call(rec, modality, rubric, client, model))
        results[key] = rs
        out[key] = [r.to_dict() for r in rs]
        n_err = sum(1 for r in rs if r.error)
        print(f"  judged {key}: {len(rs)} calls, {n_err} errors", flush=True)

    analysis = {
        "modality_narrow": J.compare_modalities(
            results.get("transcript_narrow", []), results.get("audio_narrow", [])
        ),
    }
    if "transcript_broad" in results:
        analysis["modality_broad"] = J.compare_modalities(
            results["transcript_broad"], results["audio_broad"]
        )
        analysis["rubric_transcript"] = J.compare_rubrics(
            results["transcript_narrow"], results["transcript_broad"]
        )
        analysis["rubric_audio"] = J.compare_rubrics(
            results["audio_narrow"], results["audio_broad"]
        )
    if "audio_narrow_control" in results:
        analysis["judge_identity_audio"] = J.compare_rubrics(
            results["audio_narrow"], results["audio_narrow_control"]
        )
    return {"raw": out, "analysis": analysis}


def score_all(records: list[CallRecord]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        ex = score_execution(rec)
        oc = score_outcome(rec)
        lat = latency_report(rec)
        bi = detect_barge_ins(rec)
        fr = friction_report(rec)
        rows.append(
            {
                "call_id": rec.call_id,
                "task_id": rec.task_id,
                "trial": rec.trial,
                "provider": rec.provider,
                "model": rec.model,
                "synthetic": rec.synthetic,
                "duration_s": rec.duration_s,
                "ended_reason": rec.ended_reason,
                "errors": rec.errors,
                "meta": rec.meta,
                "execution": ex.to_dict(),
                "outcome": oc.to_dict(),
                "latency": lat.to_dict(),
                "turn_latencies": [t.to_dict() for t in turn_latencies(rec)],
                "barge_in": bi.to_dict(),
                "friction": fr.to_dict(),
            }
        )
    return rows


#: Calls that ended because the *harness* ran out of road, not because the agent
#: did anything. They are reported, but never counted in a pass rate: doing so
#: would manufacture the very voice-channel degradation this study measures.
HARNESS_FAILURES = {"tts_quota_exhausted", "error", "max_call_seconds"}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_total = len(rows)
    excluded = [r for r in rows if r["ended_reason"] in HARNESS_FAILURES]
    rows = [r for r in rows if r["ended_reason"] not in HARNESS_FAILURES]
    n = len(rows)
    if not rows:
        return {
            "n_calls": 0,
            "n_calls_attempted": n_total,
            "n_excluded_harness_failure": len(excluded),
            "excluded_reasons": _counts([r["ended_reason"] for r in excluded]),
            "note": "every call ended in a harness failure; nothing is scorable",
        }
    outcome_pass = [r for r in rows if r["outcome"]["passed"]]
    scored = [r for r in rows if r["outcome"]["reward"] is not None]
    action_rewards = [
        r["execution"]["action_reward"]
        for r in rows
        if r["execution"]["action_reward"] is not None
    ]
    violations: dict[str, int] = {}
    for r in rows:
        for rule in r["execution"]["violation_rules"]:
            violations[rule] = violations.get(rule, 0) + 1

    all_eot = []
    for r in rows:
        all_eot += [
            t["end_of_turn_ms"] for t in r["turn_latencies"] if t["end_of_turn_ms"] is not None
        ]
    from voiceval.metrics.latency import percentiles

    stage_totals: dict[str, float] = {}
    for r in rows:
        for k, v in (r["latency"].get("stage_mean_ms") or {}).items():
            stage_totals[k] = stage_totals.get(k, 0.0) + v * max(1, r["latency"]["n_turns"])
    total_turns = sum(r["latency"]["n_turns"] for r in rows) or 1

    return {
        "n_calls": n,
        "n_calls_attempted": n_total,
        "n_excluded_harness_failure": len(excluded),
        "excluded_reasons": _counts([r["ended_reason"] for r in excluded]),
        "n_scored": len(scored),
        "outcome_pass_rate": (len(outcome_pass) / len(scored)) if scored else None,
        "outcome_passes": len(outcome_pass),
        "action_reward_mean": (sum(action_rewards) / len(action_rewards))
        if action_rewards
        else None,
        "clean_rate": (sum(1 for r in rows if r["execution"]["clean"]) / n) if n else None,
        "violations_by_rule": violations,
        "n_calls_with_violation": sum(1 for r in rows if r["execution"]["violation_rules"]),
        "end_of_turn_ms": percentiles(all_eot).to_dict(),
        "stage_mean_ms": {k: v / total_turns for k, v in stage_totals.items()},
        "total_turns": total_turns,
        "barge_in": {
            "attempted": sum(r["barge_in"]["n_scripted_barge_ins"] for r in rows),
            "landed": sum(r["barge_in"]["n_barge_ins"] for r in rows),
            "yielded": sum(r["barge_in"]["n_yielded"] for r in rows),
            "state_preserved": sum(r["barge_in"]["n_state_preserved"] for r in rows),
            "yield_latency_ms": percentiles(
                [
                    e["yield_latency_ms"]
                    for r in rows
                    for e in r["barge_in"]["events"]
                ]
            ).to_dict(),
        },
        "friction": {
            "long_silences": sum(r["friction"]["n_long_silences"] for r in rows),
            "overlap_total_s": sum(r["friction"]["overlap_total_s"] for r in rows),
            "agent_repeat_requests": sum(r["friction"]["agent_repeat_requests"] for r in rows),
            "caller_repeat_requests": sum(r["friction"]["caller_repeat_requests"] for r in rows),
            "agent_repetitions": sum(r["friction"]["n_agent_repetitions"] for r in rows),
            "early_terminations": sum(1 for r in rows if r["friction"]["early_termination"]),
            "caller_wer": _mean(
                [r["friction"]["caller_wer"] for r in rows if r["friction"]["caller_wer"] is not None]
            ),
        },
        "ended_reasons": _counts([r["ended_reason"] for r in rows]),
        # Protocol-level turbulence. Reported as a first-class result because it
        # bounds how much of any channel effect can be attributed to the agent:
        # a run in which the server cancelled and re-issued function calls
        # dozens of times is not a clean measurement of the model's competence.
        "protocol": {
            "server_tool_retries": sum(
                r["meta"].get("server_tool_retries", 0) for r in rows
            ),
            "duplicate_tool_calls_suppressed": sum(
                r["meta"].get("duplicate_tool_calls", 0) for r in rows
            ),
            "playout_underruns": sum(r["meta"].get("playout_underruns", 0) for r in rows),
            "unscripted_overlaps": sum(
                r["barge_in"]["n_barge_ins"] for r in rows
            ) if not any(r["barge_in"]["n_scripted_barge_ins"] for r in rows) else None,
            "harness_paused_s": round(
                sum(r["meta"].get("harness_paused_s", 0.0) for r in rows), 1
            ),
        },
    }


def _tts_spend(records: list[CallRecord]) -> dict[str, Any]:
    """TTS cost, summed from what each call recorded at the time it ran."""
    total: dict[str, int] = {}
    calls = hits = 0
    for rec in records:
        calls += int(rec.meta.get("tts_calls") or 0)
        hits += int(rec.meta.get("tts_cache_hits") or 0)
        for k, v in (rec.meta.get("tts_usage") or {}).items():
            if isinstance(v, int):
                total[k] = total.get(k, 0) + v
    return {"synthesis_calls": calls, "cache_hits": hits, "tokens": total}


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def _counts(xs):
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default="gemini_live",
                   choices=["gemini_live", "mock", "openai_realtime"])
    p.add_argument("--model", default=None, help="provider model override")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--num-tasks", type=int, default=None)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=14)
    p.add_argument("--max-call-s", type=float, default=300.0)
    p.add_argument("--tts", default="gemini")
    p.add_argument("--barge-in-turns", default=",".join(str(t) for t in DEFAULT_BARGE_IN_TURNS))
    p.add_argument("--judge-model", default=J.DEFAULT_JUDGE_MODEL)
    p.add_argument("--control-audio-model", default=J.DEFAULT_CONTROL_AUDIO_MODEL)
    p.add_argument("--no-judges", action="store_true")
    p.add_argument("--no-rubric-sweep", action="store_true")
    p.add_argument("--out", default=str(ARTIFACTS / "results.json"))
    p.add_argument(
        "--calls-dir", default=str(CALLS_DIR),
        help="where call records go. Use a separate directory per experimental "
             "arm: the barge-in arm must not be mixed into the arm compared "
             "against the text baseline, which had no interruptions.",
    )
    p.add_argument("--score-only", action="store_true",
                   help="re-score and re-judge the calls already in artifacts/calls")
    args = p.parse_args()

    ensure_registered()
    ARTIFACTS.mkdir(exist_ok=True)
    calls_dir = Path(args.calls_dir)
    calls_dir.mkdir(parents=True, exist_ok=True)
    client = GeminiClient()
    started = time.time()

    carried_judging: dict[str, Any] = {}
    carried_spend: dict[str, Any] = {}
    if args.score_only:
        records = [CallRecord.load(p) for p in sorted(calls_dir.glob("*.json"))]
        print(f"Loaded {len(records)} existing calls")
        # Re-scoring must not rewrite the run's provenance with this process's
        # defaults. The first --score-only pass did exactly that and stamped the
        # results with a barge-in plan the calls never had.
        if records:
            m = records[0].meta
            args.barge_in_turns = ",".join(
                str(t) for t in (m.get("barge_in_plan") or {}).get("turns", [])
            ) or "none"
            args.tts = m.get("tts_backend", args.tts)
            args.provider = records[0].provider
            args.model = records[0].model
        if args.no_judges and Path(args.out).exists():
            try:
                prev = json.loads(Path(args.out).read_text())
                carried_judging = prev.get("judging", {})
                carried_spend = prev.get("spend", {})
                if carried_judging:
                    print("  carrying forward existing judging (--no-judges)")
            except Exception:
                carried_judging = {}
    else:
        all_tasks = load_tasks()
        if args.tasks:
            all_tasks = [t for t in all_tasks if t.id in set(args.tasks)]
        if args.num_tasks:
            all_tasks = all_tasks[: args.num_tasks]
        print(f"Running {len(all_tasks)} tasks x {args.trials} trial(s) on {args.provider}")

        provider_kwargs: dict[str, Any] = {}
        if args.model:
            provider_kwargs["model"] = args.model
        cfg_kwargs = {
            "max_turns": args.max_turns,
            "max_call_s": args.max_call_s,
            "barge_in": build_barge_in(args.barge_in_turns),
        }
        sem = asyncio.Semaphore(args.concurrency)

        async def go():
            jobs = [
                run_one(t, trial, args.provider, provider_kwargs, args.tts, client,
                        cfg_kwargs, sem, calls_dir)
                for t in all_tasks
                for trial in range(args.trials)
            ]
            return await asyncio.gather(*jobs)

        records = [r for r in asyncio.run(go()) if r is not None]

    if not records:
        print("No calls completed; nothing to score.", file=sys.stderr)
        return 1

    print(f"\nScoring {len(records)} calls...")
    rows = score_all(records)
    scorable = [r for r in rows if r["ended_reason"] not in HARNESS_FAILURES]
    dropped = len(rows) - len(scorable)
    if dropped:
        print(f"  {dropped} call(s) excluded as harness failures, not agent failures")
    summary = aggregate(rows)

    judging: dict[str, Any] = dict(carried_judging)
    if not args.no_judges:
        print("Judging...")
        judgeable_ids = {r["call_id"] for r in scorable}
        judging = judge_all(
            [rec for rec in records if rec.call_id in judgeable_ids], client, args
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": round(time.time() - started, 1),
        "config": {
            "provider": args.provider,
            "provider_model": args.model
            or getattr(get_provider(args.provider), "model", args.provider),
            "trials": args.trials,
            "tts": args.tts,
            "judge_model": args.judge_model,
            "control_audio_model": args.control_audio_model,
            "barge_in_turns": args.barge_in_turns,
            "calls_dir": str(calls_dir),
            "max_turns": args.max_turns,
            "capabilities": get_provider(args.provider).describe(),
        },
        "summary": summary,
        "calls": rows,
        "judging": judging,
        "spend": {
            # A re-scoring pass makes no model calls, so its own counter is
            # zero; carrying the previous figure forward stops a re-score from
            # silently erasing the run's measured cost.
            "text_and_judge_tokens": (
                carried_spend.get("text_and_judge_tokens")
                if (client.usage.calls == 0 and carried_spend)
                else client.usage.to_dict()
            ),
            "tts": _tts_spend(records),
            "note": (
                "Gemini exposes no balance endpoint, so spend is reported in "
                "tokens by modality rather than in dollars I cannot verify."
            ),
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {args.out}")
    print(json.dumps(summary, indent=2, default=str)[:2500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
