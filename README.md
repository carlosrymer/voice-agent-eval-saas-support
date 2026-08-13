# voice-agent-eval-saas-support

**Try it live: [https://carlosrymer.github.io/voice-agent-eval-saas-support/](https://carlosrymer.github.io/voice-agent-eval-saas-support/)**

I took a support agent that scores 100% on a text benchmark, put it on a phone
call without changing the tasks, the policy, the tools or the grader, and
measured what happened on three axes: **Execution**, **Outcome**, **Experience**.
Then I ran the experiment the framework invites but nobody publishes — scoring
the same calls with a transcript-only judge and an audio-native judge, same
model, same rubric — to find out whether judge *modality* actually buys anything.

It does. Just not the thing I expected.

## The headline: the agreement I published did not survive a change of judge vendor

My first version of this project reported that a transcript-only judge and an
audio-native judge, given the same rubric and the same calls, agreed **exactly**
(mean |Δ| 0.00) on every property both could observe. That result was measured
with a Google judge on Google calls — same-family self-evaluation, which I
flagged at the time as the project's weakest point.

An OpenAI key later became available, so I re-ran the identical rubric, prompt
and recordings through a second vendor's judges. **The agreement did not
replicate, and the variable I had treated as a control turned out to dominate.**

Six calls, narrow rubric, identical prompts. Abstention is the judge declining to
score, which both were explicitly instructed to do when the evidence was absent:

| Judge vendor | audio-only properties |  |  | shared properties |  |  |
|---|---|---|---|---|---|---|
| | transcript abstained | audio abstained | mean \|Δ\| | transcript abstained | audio abstained | mean \|Δ\| |
| **Google** (same family as the agent) | **88.9%** | 0.0% | 1.00 | 66.7% | 5.6% | **0.50** |
| **OpenAI** (cross-vendor) | **27.8%** | 0.0% | 0.45 | 33.3% | 11.1% | **0.50** |

And the variables side by side, all on the same six calls:

| Variable changed | What was held fixed | mean \|Δ\| on a 5-point scale |
|---|---|---|
| **Judge vendor**, transcript modality | rubric, modality, calls | **2.50** |
| **Judge vendor**, audio modality | rubric, modality, calls | **1.25** |
| Rubric (narrow → broad), transcript | vendor, modality | 0.67 |
| Rubric (narrow → broad), audio | vendor, modality | 0.56 |
| Judge identity within Google (`3.6-flash` → `3.1-pro`) | rubric, modality | 0.48 |
| **Modality** (transcript → audio), shared properties | vendor, rubric | **0.50** |

Three things follow, and none of them is what I originally published.

**1. Judge vendor dominates.** Swapping the judge's vendor moved the score
**2.50** points on transcript judging and **1.25** on audio — five times and two
and a half times the effect of changing modality, and far beyond the rubric
effect that a sibling text project of mine found to dominate there. On this
evidence, an audio-judge score without a named vendor is close to
uninterpretable.

**2. Abstention discipline is a vendor property, not a modality property.** Asked
to rate pronunciation, pacing and naturalness *from a transcript* — properties
that are physically absent from one — Google's judge declined **88.9%** of the
time and OpenAI's declined only **27.8%**. Both were given the same sentence
telling them to return `null` rather than guess. So OpenAI's judge answers
questions it cannot possibly have evidence for roughly three times as often.
That is a safety-relevant difference between two judges that a single aggregate
score would completely hide.

**3. My original 0.00 was not robust.** Re-running the Google judge on the same
recordings now gives 0.50 on shared properties rather than 0.00. Combined with
the abstention instability I reported before (18/18 vs 13/18 across two identical
temperature-0 passes), the picture is consistent: **this measurement moves
between runs, and I published a single-pass value as though it were a
constant.** The direction that survives all three passes is only the weak claim —
audio judges abstain far less than transcript judges on audio-only properties.
The precise agreement figure does not survive at all.

The practical read, revised: **an audio judge does buy you coverage a transcript
judge cannot provide** — that part replicated cleanly across both vendors, with
audio abstention at 0.0% against 27.8–88.9%. But **do not treat any judged
Experience score as a stable quantity.** Pin the vendor, pin the rubric, run it
more than once, and prefer a deterministic metric wherever one exists. That is
the whole argument for how the Experience axis in this project is built, and the
cross-vendor run is the strongest evidence for it that I have.

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

**Technology:** two realtime voice-agent stacks — Gemini Live
(`bidiGenerateContent` over WebSocket) and OpenAI Realtime (GA `/v1/realtime`) —
behind one vendor-neutral seam, evaluated with LangChain's three-axis voice
framework, implemented with the emphasis deliberately inverted from judged to
measured.

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

### The provider seam — what a second vendor actually cost

The project claimed a second voice stack would be "a translation table, not a
rewrite". A key for OpenAI Realtime later arrived, so that claim got tested
rather than asserted. **Half of it held, and the half that failed is the more
useful half to know about.**

**The server-to-client translation table was exactly right.** Every event name
in `translate()` — written from published docs months before any key existed and
unit-tested only against fixture frames I authored — is confirmed by the live GA
service: `response.created`, `response.output_audio.delta`,
`response.output_audio_transcript.delta`/`.done`,
`response.function_call_arguments.done`, `response.done` with a status,
`conversation.item.input_audio_transcription.completed`, `error`. Not one needed
changing, and **nothing above the seam was edited at all** — the orchestrator,
the ledger, every Experience metric and both judges ran unmodified against a
vendor they had never seen.

**The client-to-server session shape was wrong, structurally.** The adapter was
written against the Realtime *Beta* API, which is switched off: sending
`OpenAI-Beta: realtime=v1` closes the socket immediately with
`beta_api_shape_disabled`. GA moved `modalities` → `output_modalities`, replaced
flat `input_audio_format: "pcm16"` strings with nested
`audio.input.format: {"type":"audio/pcm","rate":24000}` objects, and relocated
`voice`, `turn_detection` and `input_audio_transcription` under `audio.*`.

So the honest verdict on how comparable these two realtime protocols are: **their
event streams genuinely are interchangeable behind one normalization layer — that
is now evidence, not assertion — while their session-configuration surfaces are
not interchangeable and are not even stable across one vendor's own versions.**
That is an argument *for* the seam: all the churn stayed inside one method,
`session_update_frame`, and none of it reached the measurement layer.

**The bug that cost the most was neither.** OpenAI rejects any voice name outside
its own set, and rejects the *entire* `session.update` frame when one appears —
without closing the socket. Passing Gemini's default voice `Puck` through meant
the session silently kept its defaults: server voice-activity detection stayed
on and input transcription stayed off. A whole 16-task arm ran with empty caller
transcripts, auto-created responses fighting my explicit ones, and zero tool
calls. One invalid enum value degraded every downstream metric and nothing in the
transcript said so. The adapter now maps voices across vendors and treats a
rejected `session.update` as fatal.

Where OpenAI Realtime is *better* than Gemini Live, not merely different: it
emits `response.created` before any audio (so `emits_turn_start` is True and its
turns decompose one stage further — Gemini gives one opaque block); explicit turn
boundaries via `input_audio_buffer.commit` + `response.create` are a first-class
control path rather than a workaround for a VAD that cancels the agent's own tool
calls; and its function-call round trips were stable, needing no de-duplication
guard. Where Gemini is better: it accepts 16 kHz caller audio, and
`serverContent.interrupted` is a clearer barge-in signal than inferring one from
a `response.done` carrying status `cancelled`.

Both stacks share the failure mode that matters most for a support line:
**spelled-out identifiers**. Gemini heard `acct one zero four two` as
"ACTT1042"; OpenAI heard `p, r, i, y, a at northwindlabs dot io` as
"priyaa@northwindlabs.io" and never found the account. That is invisible to any
evaluation that does not check the *arguments* the agent passed to its tools.

Full mapping, wire notes and the procedure for adding a third stack:
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
- **Judging, Google:** 30 calls (5 passes × 6 calls), 136,835 tokens, of which
  **66,663 are audio input** — `gemini-3.6-flash` 100,526 and
  `gemini-3.1-pro-preview` 36,309.
- **Judging, OpenAI (cross-vendor control):** 12 calls (2 passes × 6 calls),
  22,549 tokens, of which 8,884 are audio input — `gpt-audio` 14,101 and
  `gpt-4.1-mini` 8,448.
- Audio judging is the most expensive line item on either vendor: 66,663 of
  Google's 90,543 prompt tokens were audio.
- **Caller brain:** 51 model calls, one per caller utterance (`gemini-3.6-flash`).
- **OpenAI speech synthesis:** billed per character rather than per token; the
  caller-voice path for the OpenAI arm is `gpt-4o-mini-tts`.
- **Dollars:** Gemini exposes no balance endpoint, so spend is reported in
  tokens by modality rather than in dollars I cannot verify.

## Honest limitations

- **No telephony.** Clean datacenter WebSocket — no jitter, packet loss, codec,
  handset or background noise. These latencies are a floor for a real PSTN call,
  not an estimate of one.
- **Single-vendor self-evaluation — now partly resolved, and worse than I
  thought.** Originally the agent, the caller's voice, the caller's brain and
  both judges were all Google models. The judging half of that confound is now
  broken: every subjective score is also computed by OpenAI judges
  (`gpt-4.1-mini` on transcript, `gpt-audio` on audio) over the identical
  recordings and rubric, and both vendors' figures are published side by side.
  Doing so did not vindicate the original numbers — it showed that judge vendor
  moves the score more than anything else I varied. What remains unresolved: the
  *agent* and the *caller* are still same-family within each arm, and there is
  still no human annotator anywhere in this project, so no subjective score here
  has ever been checked against a person.
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
