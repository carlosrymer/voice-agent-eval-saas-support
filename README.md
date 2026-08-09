# voice-agent-eval-saas-support

**Try it live: [https://carlosrymer.github.io/voice-agent-eval-saas-support/](https://carlosrymer.github.io/voice-agent-eval-saas-support/)**

I took a support agent that scores 100% on a text benchmark, put it on a phone
call without changing the tasks, the policy, the tools or the grader, and
measured what happened on three axes: **Execution**, **Outcome**, **Experience**.
Then I ran the experiment the framework invites but nobody publishes — scoring
the same calls with a transcript-only judge and an audio-native judge, same
model, same rubric — to find out whether judge *modality* actually buys anything.

It does. Just not the thing I expected.

## The headline: where the evidence is not in the transcript, the transcript judge is either silent or confidently wrong

Six calls, scored four ways: {transcript, audio} × {narrow, broad rubric}, by the
**same model** (`gemini-3.6-flash`), with the same task context and the same tool
ledger on both sides. The only difference is whether the conversation arrived as
text or as a recording. Both judges were told, in identical words, to return
`null` for anything they could not assess from what they were given.

| Criterion | in the recording only? | transcript abstained | audio abstained | both scored | mean \|Δ\| | max \|Δ\| |
|---|---|---|---|---|---|---|
| pronunciation | yes | 5/6 | 0/6 | 1 | 0.0 | 0.0 |
| pacing | yes | 4/6 | 0/6 | 2 | **2.0** | **4.0** |
| naturalness | yes | 4/6 | 0/6 | 2 | **1.0** | 2.0 |
| interruption_handling | no | 5/6 | 1/6 | 1 | 0.0 | 0.0 |
| turn_taking | no | 3/6 | 0/6 | 3 | 0.0 | 0.0 |
| overall_experience | no | 4/6 | 0/6 | 2 | 0.0 | 0.0 |

Two clean results, in opposite directions.

**Where both judges could see the evidence, they agreed exactly.** Across the
three properties a transcript can legitimately carry, every pair where both
returned a number matched to the decimal: mean absolute difference **0.00**, max
0.00. Paying for audio bought nothing there.

**Where the evidence exists only in the recording, the transcript judge mostly
knew it — and was badly wrong when it didn't.** It abstained on **13 of 18**
audio-only judgements (72%) against the audio judge's **0 of 18**. On the five
occasions it answered anyway, it disagreed with the audio judge by a mean of
**1.0** and a maximum of **4.0 points on a 5-point scale** — rating the agent's
pacing 1/5 where the audio judge, listening to it, gave 3.8/5.

So the failure mode of a transcript-only judge on an audio property is not noise.
It is a *confident* wrong answer on a minority of cases, hiding behind
well-calibrated silence on the rest. That is worse than a judge that is uniformly
noisy, because the abstentions make it look trustworthy.

Against the two controls, on the same calls:

| Variable changed | What was held fixed | mean \|Δ\| score |
|---|---|---|
| **Modality** (transcript → audio), shared properties | model, rubric, context | **0.00** |
| **Rubric** (narrow → broad), audio judge | model, modality | 0.35 |
| **Judge identity** (`gemini-3.6-flash` → `gemini-3.1-pro-preview`), audio | rubric, modality | 0.46 |
| **Modality**, audio-only properties | model, rubric, context | **1.00**, and coverage 28% → 100% |

Modality is the only one of the three that changes what can be answered at all.
For everything else, the rubric and the judge move the score more than the
modality does — which extends a finding from a sibling text project of mine,
where swapping the judge three ways moved a headline by 0% while the rubric moved
it from 0% to 100%.

The practical read on "should I pay for an audio judge": **only for properties a
transcript cannot contain** — and there, pay for it, because the cheap
alternative's errors are large and confident. It cost 66,663 audio input tokens
across 30 judging calls to establish that.

### The caveat that undercuts all of this, which I am reporting because it is the point

I ran the identical judging pass twice, at `temperature = 0.0`, over the same six
recordings. The abstention counts were **not the same**: the first pass abstained
on 18 of 18 audio-only judgements, the second on 13 of 18. The scores where both
passes answered were stable; *whether the judge answered at all* was not.

Everything above is from the second pass, which is the one committed to
`artifacts/results_main.json`. The direction of the result is robust — the
transcript judge abstains far more than the audio judge, and disagrees sharply
when it doesn't — but the exact abstention rate is not a stable quantity at this
sample size, and anybody publishing one as if it were (including me, if I stopped
at one pass) would be publishing a coin flip. This is the strongest argument in
the project for keeping Experience mostly on the deterministic side: the measured
numbers re-derive bit-identically, and this one does not.

## The channel comparison, with the caveat that dominates it

Same 16-task domain, same 7-rule policy, same 18 tools, same τ² evaluators as my
published text run. I ran 6 of those tasks — chosen to exercise all seven policy
rules — over Gemini Live.

| | Text baseline (published) | This voice run |
|---|---|---|
| Agent | `gemini-3.6-flash` | `gemini-3.1-flash-live-preview` |
| Tasks × trials | 16 × 4 | **6 × 1** |
| Outcome pass (env assertions) | 100% | **33.3% (2/6)** |
| Policy violations | 0 | **2** (P6 cross-account, P7 API key) |
| Median end-of-turn latency | n/a (text) | **9.3 s** (P95 31.1 s) |

**I do not think that 33% is a clean measurement of the model, and the run says
so itself.** The same run recorded:

- **29 server-side tool-call cancellations** — Gemini Live issuing a function
  call and then abandoning it, across 6 calls.
- **18 duplicate tool calls suppressed** by the harness. Without that guard the
  writes would have applied twice and manufactured a P2 "split the credit"
  policy violation out of a transport retry.
- **28 unscripted overlaps** — the caller talking over the agent because the
  agent's response arrived after the harness had already handed the floor back.
- **43 stretches of dead air** over 1 second, and 53.2 s of simultaneous speech.

A 9.3-second median response is, by itself, enough to wreck a conversation: the
simulated caller repeatedly asked "Hello? Are you still there?" and three of six
calls ended early. What this number measures is **the whole voice stack as I was
able to drive it**, not the model's reasoning ability. The text arm had none of
this turbulence because a text channel has no turn-taking to get wrong.

That is a real and useful result — "the channel is where the failures are" is
worth knowing — but it is not "the model got dumber on the phone", and I am not
going to write it up as if it were.

Where the latency went, per agent turn (19 turns; stages sum to the total by
construction, with anything unattributable reported as residual rather than
folded into a neighbour):

| Stage | Bounded by | Mean ms/turn |
|---|---|---|
| `asr_ms` | caller speech end → final ASR | 1,711 |
| `to_first_audio_ms` | → first audio byte | 7,363 |
| residual | — | 0 |

Gemini Live emits no "response started" frame, so inference and speech synthesis
cannot be separated for this provider — the harness reports one honest 7.4 s
block rather than an invented breakdown. This is exactly what the capability
descriptors exist for.

## What this showcases

**Technology:** the realtime voice-agent stack — Gemini Live
(`bidiGenerateContent` over WebSocket, server-driven barge-in, native tool
calling) — evaluated with LangChain's three-axis voice framework, implemented
with the emphasis deliberately inverted.

### Experience is measured, not judged

The article reaches for an LLM judge whenever an Experience property is hard to
compute. I had a specific reason not to: the only realtime voice stack available
to me is Gemini and the only audio-capable judge available to me is Gemini, so a
judged Experience score is a Gemini agent graded by a Gemini judge. That is
same-family self-evaluation and it is a real credibility problem.

So almost all of Experience here is computed from the recordings and the event
timeline — latency percentiles per stage, barge-in detection with yield latency
and state-loss checks, dead air, overlapping speech, clarification and repeat
loops, self-repetition, turn cadence, and caller word error rate (free, because
the harness authored every caller utterance before speaking it: **14.4%**). The
judges are confined to pronunciation, pacing and naturalness, and every judged
number on the site is labelled single-family.

**All of that measurement is verified against synthetic audio with known ground
truth.** 150 tests, of which the ones that matter assert against numbers I chose
rather than numbers the code produced: a 1.400 s silence gap is recovered as
1.38 s, a 300 ms overlap as 320 ms, a speech burst placed at 1.234 s comes back
within one analysis frame. Every detector has a paired near-miss — a caller
starting 80 ms *after* the agent stops must not count as a barge-in; an ordinary
service question must not count as a clarification loop. A violation detector
that silently never fires reports a clean 0% and looks like good news.

### What surprised me about Gemini Live

It was much harder to drive correctly than the documentation suggests, and every
failure produced *plausible output* rather than an error. In order of how long
each cost me:

1. **Automatic voice-activity detection cancelled the agent's own tool calls.**
   At the end of every caller utterance the server re-detected activity and
   cancelled the in-flight function call ~190 ms after issuing it. No tool-using
   task could ever complete. The fix is
   `realtimeInputConfig.automaticActivityDetection.disabled` plus explicit
   `activityStart`/`activityEnd` — which removes the server's endpointing delay
   from the measured latency, stated wherever latency appears.
2. **The previous turn's `turnComplete` and final transcript are flushed when
   new caller audio arrives.** Treat either as belonging to the current turn and
   every turn ends instantly wearing the *previous* turn's words. My first live
   run produced eight agent turns with identical text and I nearly believed it.
3. **The server cancels and re-issues identical function calls** (29 times in 6
   calls). Execute both and the write applies twice.
4. **`realtimeInput.mediaChunks` closes the socket** with a 1007 and a
   deprecation string. Most third-party examples still show it; the live field
   is `realtimeInput.audio`.
5. **Streaming caller audio at speaking pace** let the server start a response
   mid-utterance and then discard it. The harness now spends the utterance's
   duration in real time and transmits in one burst, which biases the latency
   *pessimistic* — the safer direction.

**The honest verdict on the technology:** the realtime API works and the audio is
good, but a production deployment lives or dies on the turn-taking layer, not on
the model. Four of my five worst bugs were protocol handling, none of them threw
an exception, and all of them produced transcripts that read fine. If you are
evaluating one of these stacks, budget for that and instrument it — which is
what this repo is.

### The provider seam

A second voice stack is expected on this project, so the measurement layer is
written against a vendor-neutral contract, not against Gemini. Timestamps are
stamped on receipt before parsing (vendors report their own timings
inconsistently; all of them put bytes on a socket). Capabilities are declared
data, so a metric that cannot be computed for a provider reports `None` with a
reason instead of a number meaning something different from the one beside it.

Three adapters exist: `gemini_live` (wire-verified), `mock` (a deterministic
realtime simulator on a virtual clock, which is what lets the whole pipeline run
offline with no key), and `openai_realtime` — whose pure `translate()` function
is unit-tested against hand-written OpenAI-shaped frames but has **never touched
the wire**, because there is no key. `wire_verified = False`, `connect()` refuses
without credentials, and the flag propagates into any report including it. See
[PROVIDERS.md](PROVIDERS.md).

## The use case

A B2B SaaS support line for "Loopline", a marketing-automation platform: account
credits with an escalation threshold, seat changes that may not take effect
mid-cycle, sending limits gated on domain verification and plan ceilings,
cross-account confidentiality, and a rule against ever asking for an API key.
Two databases, 18 agent tools, 7 customer-only tools, 16 authored tasks.

It is a fair test rather than a flattering one because **it already has a public
text baseline scored by the same code.** Three of the seven policy rules leave no
trace in the final database — an agent can resolve the request perfectly and
still have written to an account before verifying identity, read another
company's record, or asked for a secret. Those are caught by an action-ledger
audit, not by the reward. On this voice run that audit caught two violations the
Outcome score would have missed entirely, including one on a task the agent
otherwise passed.

## Docs

- [Architecture](ARCHITECTURE.md) — system design, components, data flow, deployment
- [PRD](PRD.md) — problem statement, scope, success criteria
- [Providers](PROVIDERS.md) — the seam, the Gemini wire notes, and how to add a stack

## Running locally

```bash
# 1. Install (Python 3.12+, uv)
uv sync --group dev

# 2. The full test suite — no API key, no spend, ~4 minutes.
#    Includes an end-to-end run of the whole pipeline on the mock provider.
uv run pytest -q

# 3. Re-derive every published number from the committed artifacts.
#    No API key, no spend.
uv run python scripts/verify_numbers.py

# 4. View the site (reads committed JSON over HTTP)
python3 -m http.server 8000 --directory site   # -> http://localhost:8000

# 5. Re-run the experiment. Costs money and overwrites artifacts/.
#    Requires GEMINI_API_KEY.
docker run -d -p 6006:6006 arizephoenix/phoenix:latest   # optional trace UI
uv run python -m voiceval.run_experiment \
  --provider gemini_live --trials 1 --concurrency 3 \
  --tasks T01_credit_within_cap T02_credit_over_cap_escalate \
          T05_seat_reduction_schedule T09_send_limit_unverified_refuse \
          T12_cross_account_disclosure T13_api_key_never_shared \
  --barge-in-turns none \
  --calls-dir artifacts/calls_main --out artifacts/results_main.json
uv run python -m voiceval.report --results artifacts/results_main.json
```

## Committed artifacts

Everything published is re-derivable from these, and
`scripts/verify_numbers.py` proves it offline:

| File | Contents |
|---|---|
| `artifacts/results_main.json` | Every computed number: three axes per call, per-turn latency, judging |
| `artifacts/calls_main/*.json` | Per call: normalized event stream, action ledger, both transcripts, metadata |
| `artifacts/calls_main/*.wav` | Both audio tracks per call at full fidelity, on one session clock |
| `artifacts/otel_spans.jsonl` | OpenTelemetry spans |
| `site/data/*.json` | Everything the site renders, generated by `voiceval/report.py` |

The recordings are committed (69 MB) specifically so the Experience numbers
reproduce *exactly* rather than approximately — a voice-activity detector re-run
over the same samples returns the same boundaries.

No number on the site is typed in by hand.

## Stack

Python 3.12 · `uv` · τ²-bench (`sierra-research/tau2-bench` v1.0.1, used as a
dependency for the domain *and* its evaluators) · Gemini Live
`gemini-3.1-flash-live-preview` (agent) · Gemini TTS (caller voice) ·
`gemini-3.6-flash` (caller brain, both judges) · `gemini-3.1-pro-preview`
(judge-identity control) · `numpy` · `websockets` · OpenTelemetry ·
Arize Phoenix (self-hosted, optional) · static site in hand-rolled HTML/CSS/SVG,
no libraries.

There is no LangSmith key in this environment, so the tracing half of the
article's workflow runs on a self-hosted backend instead. That is a substitution
of tool, not of method — the spans use OpenInference conventions and the method
being demonstrated is the three-axis evaluation, not any particular UI.

## What it cost

Measured, not estimated, for the 6-call run:

- **Speech synthesis:** 46 calls, 8,740 audio output tokens, 5 cache hits.
  Spread across three TTS models because Gemini meters
  `generate_requests_per_model_per_day` **per model** at 100/day — a limit I hit
  mid-experiment, which is why the harness now fails over across models and
  excludes quota-truncated calls from scoring as *harness* failures rather than
  agent failures.
- **Judging:** 30 calls (5 passes × 6 calls), 135,899 tokens, of which **66,663
  are audio input**. Audio judging is the single most expensive line item here.
- **Caller brain:** 51 model calls, one per caller utterance.
- **Dollars:** Gemini exposes no balance endpoint, so spend is reported in
  tokens by modality rather than in dollars I cannot verify.

## Honest limitations

- **No telephony.** Clean datacenter WebSocket — no jitter, packet loss, codec,
  handset or background noise. These latencies are a floor for a real PSTN call,
  not an estimate of one.
- **Single-vendor self-evaluation.** The agent, the caller's voice, the caller's
  brain and both judges are all Google models. Every subjective score inherits
  that, which is the whole reason Experience is mostly measured here.
- **Small sample, stated exactly:** 6 tasks × 1 trial = 6 calls, 19 agent turns
  with a latency observation, 18 criterion-judgements per modality arm. This is a
  method demonstration with real numbers, not a benchmark.
- **The channel comparison is confounded** by 29 server-side tool cancellations
  and 28 unscripted overlaps, as described above. Directional, not clean.
- **Turn boundaries are signalled explicitly**, so the measured latency excludes
  the server's own endpointing delay — a production deployment on automatic VAD
  adds that on top.
- **The caller is a simulation** — a text model whose lines a TTS model reads.
  Cleaner and more predictable than a real customer, which flatters the 14.4%
  word error rate.
- **Barge-in was not exercised in this arm.** Interruptions are scripted, and I
  ran the compared arm without them so it matched the text baseline, which had
  none. The barge-in machinery is implemented and tested against fixtures; the
  28 overlaps reported here are unscripted turbulence, not tests.
- **Phrase-matched friction is recall-limited.** The regex list is published and
  the share of agent questions it failed to classify is reported alongside.
- **The OpenAI Realtime adapter has never been run against the service.**

## Deployed via

GitHub Pages, served from the `gh-pages` branch. The publishing token was not
granted GitHub's `workflow` scope, so no Actions workflow could be installed and
the branch is pushed by hand with `scripts/deploy_pages.sh`; the workflow that
would have done it is parked at `deploy/github-pages-workflow.yml`. Nothing in
this repo runs automatically, and this says so rather than describing CI that is
not there.

---
Part of an ongoing series of small, real-world builds trialing frontier AI models,
frameworks, and tools as they ship.
