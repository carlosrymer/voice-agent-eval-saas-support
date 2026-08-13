# PRD — voice-agent-eval-saas-support

## Problem statement

Voice agents are being shipped into customer support faster than anyone has
agreed how to evaluate them. LangChain's article *How to evaluate voice agents:
execution, outcomes, and experience* proposes a sensible three-axis framework,
and like most eval writing it reaches for an LLM judge whenever a property is
hard to compute.

That creates two gaps worth attacking.

**First, nobody has published what happens to a *known* agent when you move it
from text to voice.** Text agent benchmarks are mature — τ²-bench, pass^k,
policy auditing. Voice evaluations start from scratch with new domains and new
tasks, so any difference could be the domain, the tasks, or the channel. If you
hold the domain, the tasks, the policy, the tools *and the evaluator* fixed and
change only the channel, the difference is the channel.

**Second, "Experience" is treated as a judging problem when much of it is a
measurement problem.** Latency, interruption handling, dead air, overlapping
speech, repetition, turn-taking cadence and speech-recognition accuracy are all
computable from the recordings and the event timeline. Asking a model to rate
them is slower, costlier, unreproducible, and — in this environment — circular,
because the only available audio judge is from the same family as the only
available voice agent.

## Target user

Someone about to put a voice agent in front of customers who needs to know
what to measure and what it costs to measure it — and someone evaluating a
realtime voice API who wants to know what actually happens when you drive it
hard, rather than what the quickstart shows.

## Goals

- Port a shipped τ²-bench B2B SaaS support domain to voice **without changing
  the tasks, the policy, the tools or the evaluators**, so the text baseline
  stays a valid comparison.
- Implement all three axes: Execution, Outcome, Experience.
- Make Experience **mostly measured, not judged**, and say exactly which parts
  are which.
- Test the measurement layer against synthetic audio with known ground truth, so
  the detectors are trustworthy before they are pointed at real calls.
- Run the headline experiment: **does judge modality matter?** Score the same
  calls transcript-only and audio-native, with the same model and rubric, and
  measure where and why they disagree.
- Build a pluggable provider seam so a second voice stack is an extension rather
  than a rewrite.
- Be explicit about every limitation, especially the ones that flatter the
  results.

## Non-goals

- **Telephony.** No PSTN, no jitter, no codecs. Out of scope and stated as such;
  the latency numbers are a floor, not a production estimate.
- **A leaderboard.** One agent, one domain, a small stated sample. This is a
  method demonstration with real numbers attached, not a benchmark.
- **Beating the text baseline.** The point is to measure the difference
  honestly, whichever way it goes.
- **A hosted service.** Static site, committed artifacts, reproducible locally.
- **Human evaluation.** No annotators were available, so no claim rests on human
  preference — which is precisely why the subjective residue is kept small and
  labelled.

## Scope (MVP)

1. **Provider seam** — `VoiceProvider`/`VoiceSession` with a normalized event
   stream, transport-stamped timestamps and declared capabilities. Gemini Live
   (verified), OpenAI Realtime (translation layer only, no key), and a
   deterministic mock.
2. **Caller simulator** — τ² voice guidelines + task persona + customer-side
   tools, driving TTS, with a scripted barge-in plan.
3. **Measured Experience** — energy VAD, end-of-turn latency with a stage
   decomposition that always sums to the total, barge-in detection with yield
   latency and state-loss checks, dead air, overlap, clarification and repeat
   loops, self-repetition, turn cadence, caller word error rate.
4. **Execution + Outcome** — τ²'s own `ActionEvaluator` and
   `EnvironmentEvaluator` on a reconstructed trajectory, plus the ported
   seven-rule policy auditor.
5. **Judges** — transcript-only vs audio-native, same model, plus a narrow/broad
   rubric sweep and a second-model control.
6. **Fixture test suite** — synthetic audio with declared ground truth, plus a
   fully offline end-to-end run.
7. **One command** for the whole experiment; a static site; committed artifacts.

## User stories

- As an engineer shipping a voice agent, I want latency broken down per stage,
  so I know whether to optimise the model, the tools or the speech synthesis.
- As an engineer, I want to know whether my agent yields when interrupted and
  whether it loses its place afterwards, because a transcript cannot tell me.
- As someone choosing an eval strategy, I want to know whether paying for an
  audio judge buys anything over a transcript judge, and on which properties.
- As a reviewer, I want to hear the call behind any number I doubt.
- As a maintainer, I want a second voice vendor to be a translation table, so
  numbers stay comparable across vendors.
- As a sceptic, I want every detector to have a test proving it fires on a real
  violation *and* stays quiet on a near-miss.

## Success criteria

**The technology being trialled is the realtime voice-agent stack — Gemini Live
and, later, OpenAI Realtime — plus the three-axis evaluation method.** The claim under test is that a realtime
voice API can be driven as a tool-using, policy-bound support agent and
evaluated with the same rigour as a text agent.

Success means:

1. The fixture suite passes and each detector has a fire *and* a near-miss case.
   — **Met**: 146 tests, including exact boundary recovery from synthetic audio
   and an offline end-to-end run of the whole pipeline.
2. The voice arm runs the identical tasks through the identical evaluators as
   the published text baseline. — **Met**.
3. Experience is majority-measured, with the judged residue named. — **Met**:
   everything except pronunciation, pacing and naturalness is computed.
4. The judge-modality experiment produces a decidable answer, including
   abstention behaviour. — **Met, and the answer overturned the first one.**
   With a second vendor's judges available, the transcript-vs-audio agreement I
   originally published (mean |Δ| 0.00) did not replicate, and judge *vendor*
   turned out to move the score five times more than modality does. Reported as
   the headline rather than buried, because a negative result about one's own
   published number is the most useful thing this project produced.
5. A second realtime stack drops in behind the seam without touching the
   measurement layer. — **Half met, honestly.** The server-to-client translation
   table needed no changes and nothing above the seam was edited; the
   client-to-server session shape had to be rewritten because it targeted an API
   version that has since been switched off.
5. Limitations are stated where the numbers are, not only in an appendix.
   — **Met**.

**Where the technology fell short**, plainly: Gemini Live was considerably
harder to drive correctly than its documentation suggests. Automatic
voice-activity detection cancelled the agent's own in-flight tool calls at the
end of every caller utterance, so no tool-using task could complete until turn
boundaries were signalled explicitly. The previous turn's `turnComplete` and
final transcript are flushed when new caller audio arrives, which silently gives
every turn the previous turn's words unless you guard for it. The server
sometimes cancels and re-issues an identical function call, which double-applies
writes unless de-duplicated. And `realtimeInput.mediaChunks`, which most
examples still show, closes the socket. None of these are exotic edge cases —
they are what happens on the first real conversation — and every one of them
produced plausible-looking output rather than an error.

## Risks / open questions

| Risk | Mitigation |
|---|---|
| Single-vendor self-evaluation on subjective scores | **Partly resolved.** Every subjective score is now computed by both a Google and an OpenAI judge over identical recordings and rubrics, and both are published. What remains: agent and caller are still same-family within an arm, and no human has ever checked a subjective score here |
| Judged scores treated as stable quantities | Three passes showed they are not; every judged figure is published with its vendor, its rubric and its run-to-run variation, and the deterministic metrics are the ones load-bearing |
| Small sample | Counts stated exactly everywhere; no claim of benchmark status |
| Daily per-model API quotas truncating a run | TTS fails over across three models with independent daily budgets; quota-truncated calls are excluded from scoring as *harness* failures, never counted as agent failures |
| Harness bugs read as agent failures | Offline end-to-end test over the real pipeline; harness-failure reasons excluded from every rate and reported separately |
| Latency artefacts from the test rig | Rig thinking time is excluded from the timeline and reported; audio transmission is burst-then-measure, which biases latency pessimistic rather than optimistic |
| Detectors that never fire | Every detector has a paired near-miss test |
| Barge-in confounding the channel comparison | Barge-in runs as a separate arm; the arm compared to text has no interruptions, matching the baseline |

Open: whether the same pattern holds with real telephony; whether a
cross-vendor judge changes the modality result; whether audio judging pays off
at larger sample sizes.

## Timeline

One working session: substrate reuse and provider seam, measurement layer and
its fixture tests, live integration and the four wire-level defects it exposed,
the funded run, scoring, site and docs.
