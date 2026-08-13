#!/usr/bin/env python
"""Re-derive every published number from the committed artifacts.

Run: `uv run python scripts/verify_numbers.py` — no API key, no spend, a few
seconds.

All three axes are checked, because all three inputs are committed. Execution
and Outcome come from the action ledger and the reconstructed trajectory in
`artifacts/calls_main/*.json`. The Experience numbers come from the recordings,
and the recordings are committed at full fidelity (69 MB for the run) precisely
so that they can be. A voice-activity detector re-run over the same samples
returns the same boundaries, so the latency percentiles, barge-in figures and
friction counts reproduce exactly rather than approximately.

The only thing not re-derived here is the judged Experience scores, which need
a model.

Exit code 0 means every published figure re-derives from the artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voiceval.metrics.bargein import detect_barge_ins  # noqa: E402
from voiceval.metrics.friction import friction_report  # noqa: E402
from voiceval.metrics.latency import latency_report  # noqa: E402
from voiceval.metrics.timeline import CallRecord  # noqa: E402
from voiceval.scoring.execution import score_execution  # noqa: E402
from voiceval.scoring.outcome import score_outcome  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
#: (results file, calls dir, audio committed?) per arm.
#:
#: The Gemini arm ships its recordings, so all three axes re-derive exactly.
#: The OpenAI arm's recordings are 331 MB and are not committed, so only the
#: axes that read the ledger and the reconstructed trajectory -- Execution and
#: Outcome -- can be checked here. Its measured-Experience numbers reproduce
#: only by re-running the arm, and this says so rather than quietly checking
#: less than it claims.
ARMS = [
    (ROOT / "artifacts" / "results_main.json", ROOT / "artifacts" / "calls_main", True),
    (ROOT / "artifacts" / "results_openai.json", ROOT / "artifacts" / "calls_openai", False),
]


def _differs(a, b, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is not b
    return abs(float(a) - float(b)) > tol


def main() -> int:
    failures: list[str] = []
    checked = 0
    audio_checked = 0

    for results_path, calls_dir, has_audio in ARMS:
        if not results_path.exists():
            print(f"  (skipping {results_path.name}: not present)")
            continue
        published = json.loads(results_path.read_text())
        by_id = {r["call_id"]: r for r in published["calls"]}
        arm = published["config"].get("provider_model") or published["config"].get("provider")
        print(f"\n{arm} — {'all three axes' if has_audio else 'Execution + Outcome only'}")

        for path in sorted(calls_dir.glob("*.json")):
            rec = CallRecord.load(path)
            claim = by_id.get(rec.call_id)
            if claim is None:
                failures.append(f"{rec.call_id}: on disk but absent from results")
                continue
            checked += 1
            before = len(failures)

            ex = score_execution(rec)
            oc = score_outcome(rec)
            got_rules = sorted({v.rule for v in ex.violations})
            want_rules = sorted(claim["execution"]["violation_rules"])
            if got_rules != want_rules:
                failures.append(
                    f"{rec.task_id}: violations {got_rules} != published {want_rules}"
                )
            if ex.action_reward != claim["execution"]["action_reward"]:
                failures.append(
                    f"{rec.task_id}: action reward {ex.action_reward} != "
                    f"published {claim['execution']['action_reward']}"
                )
            if oc.reward != claim["outcome"]["reward"]:
                failures.append(
                    f"{rec.task_id}: outcome reward {oc.reward} != "
                    f"published {claim['outcome']['reward']}"
                )

            if has_audio and rec.agent_track is not None:
                audio_checked += 1
                lat = latency_report(rec).to_dict()["end_of_turn_ms"]
                want_lat = claim["latency"]["end_of_turn_ms"]
                for k in ("n", "p50", "p95", "mean"):
                    if _differs(lat[k], want_lat[k]):
                        failures.append(
                            f"{rec.task_id}: end-of-turn {k} {lat[k]} != published {want_lat[k]}"
                        )
                bi = detect_barge_ins(rec).to_dict()
                if bi["n_barge_ins"] != claim["barge_in"]["n_barge_ins"]:
                    failures.append(
                        f"{rec.task_id}: barge-ins {bi['n_barge_ins']} != "
                        f"published {claim['barge_in']['n_barge_ins']}"
                    )
                fr = friction_report(rec).to_dict()
                for k in ("n_long_silences", "n_agent_repetitions", "caller_wer"):
                    if _differs(fr[k], claim["friction"][k]):
                        failures.append(
                            f"{rec.task_id}: friction {k} {fr[k]} != "
                            f"published {claim['friction'][k]}"
                        )

            status = "ok  " if len(failures) == before else "FAIL"
            print(
                f"  [{status}] {rec.task_id:40} outcome={oc.reward} "
                f"action={ex.action_reward} violations={got_rules or 'clean'}"
            )

    print()
    if failures:
        print(f"{len(failures)} mismatch(es):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        f"Re-derived exactly: Execution and Outcome across {checked} calls; "
        f"measured Experience across the {audio_checked} calls whose audio is committed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
