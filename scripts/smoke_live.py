import asyncio, sys, json
from voiceval.domain import task_by_id, ensure_registered
from voiceval.providers.gemini_live import GeminiLiveProvider
from voiceval.tts import GeminiTTS
from voiceval.llm import GeminiClient
from voiceval.orchestrator import run_call, CallConfig
from voiceval.caller.simulator import BargeInPlan

async def main():
    ensure_registered()
    task = task_by_id(sys.argv[1] if len(sys.argv) > 1 else "T01_credit_within_cap")
    provider = GeminiLiveProvider()
    tts = GeminiTTS()
    client = GeminiClient()
    cfg = CallConfig(max_turns=8, max_call_s=240, barge_in=BargeInPlan(turns=(1,), offset_s=0.8))
    rec = await run_call(task, provider, tts, client, cfg)
    print("ended:", rec.ended_reason, "| dur", round(rec.duration_s,2), "s")
    print("caller utts:", len(rec.caller_utterances), "agent utts:", len(rec.agent_utterances))
    print("tools:", [(t.name, t.ok) for t in rec.tool_executions])
    print("errors:", rec.errors)
    print("underruns:", rec.meta.get("playout_underruns"))
    for r in rec.transcript()[:24]:
        print(f"  [{r['t']:6.2f}] {r['role']:6} {str(r['text'])[:110]}")
    print("caller track", round(rec.caller_track.duration_s,2), "agent track", round(rec.agent_track.duration_s,2))
    from voiceval.metrics.latency import latency_report
    from voiceval.metrics.bargein import detect_barge_ins
    print("latency:", json.dumps(latency_report(rec).to_dict()["end_of_turn_ms"]))
    print("bargein:", json.dumps({k:v for k,v in detect_barge_ins(rec).to_dict().items() if k!='events'}))
    print("client usage:", client.usage.to_dict())
    print("tts calls:", tts.n_calls, "cache hits:", tts.n_cache_hits, tts.usage)
    rec.save("artifacts/smoke")

asyncio.run(main())
