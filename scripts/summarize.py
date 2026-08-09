#!/usr/bin/env python
"""Print the numbers the README quotes, straight from artifacts/results_*.json."""
import json, sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/results_main.json")
d = json.loads(p.read_text())
s, c = d["summary"], d["config"]
print(f"=== {p} ===")
print(f"provider {c['provider_model']} | trials {c['trials']} | barge-in {c['barge_in_turns']}")
print(f"attempted {s.get('n_calls_attempted')} | scored {s['n_calls']} | "
      f"excluded(harness) {s.get('n_excluded_harness_failure')} {s.get('excluded_reasons')}")
print(f"OUTCOME pass {s['outcome_passes']}/{s['n_scored']} = {s['outcome_pass_rate']}")
print(f"EXEC action_reward_mean {s['action_reward_mean']} | clean_rate {s['clean_rate']}")
print(f"VIOLATIONS {s['violations_by_rule']} | calls with any: {s['n_calls_with_violation']}")
print(f"EOT ms {json.dumps(s['end_of_turn_ms'])}")
print(f"STAGES {json.dumps({k: round(v,1) for k,v in s['stage_mean_ms'].items()})}")
print(f"TURNS {s['total_turns']}")
print(f"BARGE-IN {json.dumps({k:v for k,v in s['barge_in'].items() if k!='yield_latency_ms'})}")
print(f"  yield_latency {json.dumps(s['barge_in']['yield_latency_ms'])}")
print(f"FRICTION {json.dumps(s['friction'])}")
print(f"ENDED {json.dumps(s['ended_reasons'])}")
print(f"SPEND {json.dumps(d['spend']['text_and_judge_tokens'])}")
j = d.get("judging", {}).get("analysis", {})
mn = j.get("modality_narrow")
if mn:
    print(f"\n--- JUDGE MODALITY (narrow rubric), n={mn['n_calls']} calls")
    for r in mn["per_criterion"]:
        print(f"  {r['criterion']:22} audio_only={str(r['audio_only_property']):5} "
              f"t={r['mean_transcript']} a={r['mean_audio']} "
              f"|d|={r['mean_abs_delta']} abst t/a={r['transcript_abstentions']}/{r['audio_abstentions']} "
              f"of {r['n_pairs']}")
    print("  summary:", json.dumps(mn["summary"], indent=None))
for k in ("rubric_transcript", "rubric_audio", "judge_identity_audio", "modality_broad"):
    if k in j:
        v = j[k]
        print(f"  {k}: mean|d|={v.get('mean_abs_delta')} n={v.get('n_calls')}")
print("\nPER CALL:")
for r in d["calls"]:
    print(f"  {r['task_id']:38} {r['ended_reason']:20} outcome={r['outcome']['passed']} "
          f"act={r['execution']['action_reward']} viol={r['execution']['violation_rules']} "
          f"eot_p50={r['latency']['end_of_turn_ms']['p50']} turns={r['latency']['n_turns']} "
          f"dup_tools={r['meta'].get('duplicate_tool_calls',0)} "
          f"retries={r['meta'].get('server_tool_retries',0)}")
