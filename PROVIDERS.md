# Adding a voice provider

Everything above `voiceval/providers/base.py` — the orchestrator, the action
ledger, every Experience metric, both judges, the report, the site — is written
against one normalized contract and has no idea which vendor produced a call.
Adding a stack means writing a translation layer, not touching the measurement
layer.

I did not build the seam because abstraction is nice. I built it because a
second realtime stack (OpenAI Realtime) is expected on this project later, and
if the harness were written against Gemini's wire format then every number
measured before that arrives would be incomparable with every number measured
after.

## The contract

```python
class VoiceProvider:
    name: str
    wire_verified: bool                      # has this ever run against the live service?
    def capabilities(self) -> ProviderCapabilities: ...
    async def connect(self, config: SessionConfig, clock: Clock) -> VoiceSession: ...

class VoiceSession:
    async def send_audio(self, pcm: PCM, *, ground_truth_text: str | None = None): ...
    async def commit_turn(self): ...                       # close the caller's turn
    async def send_tool_result(self, call_id, name, payload): ...
    async def cancel_response(self): ...                   # stop speaking now
    def events(self) -> AsyncIterator[ServerEvent]: ...
    async def close(self): ...
```

Three rules make the numbers comparable across vendors.

**1. Stamp time on receipt, not from vendor telemetry.** Every `ServerEvent`
carries `t`, seconds since the session opened, taken the instant the frame comes
off the socket and *before* it is parsed. Vendors report their own timings
inconsistently or not at all; the one thing they all do identically is put bytes
on a socket. A cross-vendor latency comparison that trusts vendor timings is
comparing telemetry, not speed.

**2. Declare capabilities; never assume them.** Realtime stacks differ in ways
that change what a metric means. `ProviderCapabilities` makes those differences
data, and any metric that cannot be computed for a provider reports `None` with
a reason instead of a number that means something different from the one beside
it.

**3. PCM in, PCM out, sample rate attached.** Base64, protobuf, Opus, container
formats — all of that stays inside the adapter.

## What the flags actually change

| Capability | Effect when false |
|---|---|
| `emits_caller_transcript` | `asr_ms` is `None`; that time becomes part of a later stage or the residual. No word-error-rate either. |
| `emits_turn_start` | `to_turn_start_ms` is `None`; inference and speech synthesis cannot be separated, and the whole gap shows as `to_first_audio_ms`. |
| `emits_interrupt_event` | Barge-in is still measured — from the audio — but `signal_latency_ms` is `None`. |
| `server_barge_in` | The harness must call `cancel_response()` to emulate it, and the run manifest records that it did. |
| `server_turn_detection` | The harness holds the line open with silence so the server can endpoint; with manual signalling it does not. |

The latency report is built so this stays honest: named stages plus
`unattributed_ms` always sum to the measured total, so a provider that reports
fewer markers gets a bigger residual rather than a flatteringly small breakdown.

## The two adapters that exist

### `gemini_live` — verified against the live service

Wire facts, all confirmed by running them, not by reading docs:

- `wss://…/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=…`
- First client frame `setup`; server replies `setupComplete`.
- Caller audio: `realtimeInput.audio` (a `{mimeType, data}` blob).
  **`realtimeInput.mediaChunks`, which most third-party examples still show, is
  rejected** — the socket closes with 1007 and a deprecation string rather than
  an error frame.
- Agent audio: `serverContent.modelTurn.parts[].inlineData`, `audio/pcm;rate=24000`.
- Tools: `toolCall.functionCalls[]` → `toolResponse.functionResponses[]`, same id.
- Barge-in: `serverContent.interrupted`; cancelled tool calls arrive separately
  as `toolCallCancellation`.
- No distinct "response started" frame, hence `emits_turn_start=False`.

Two behaviours cost me most of the debugging time on this project, and both are
worth knowing before you write against this API:

- **Automatic voice-activity detection cancelled the agent's own tool calls.**
  At the end of every caller utterance the server re-detected activity and
  cancelled the in-flight function call about 190 ms after issuing it, so no
  tool-using task could ever complete. The fix is
  `realtimeInputConfig.automaticActivityDetection.disabled` plus explicit
  `activityStart` / `activityEnd`. It costs the server's endpointing delay from
  the measured latency, which is stated wherever latency is reported.
- **The previous turn's `turnComplete` and final transcript are flushed when new
  caller audio arrives.** Treating either as belonging to the current turn ends
  every turn instantly, wearing the previous turn's words. Only audio or a tool
  call proves a frame belongs to the turn you are draining.

### `openai_realtime` — wire-verified (GA `/v1/realtime`)

Real frames round-trip: session configuration, caller audio in, speech
transcription, a function call out, a tool result back in, agent audio out.

**The seam claim held on one side and failed on the other, and the split is the
useful part.**

*The server-to-client translation table was exactly right.* Every event name in
`translate()` — written from published documentation months before a key
existed, and unit-tested only against fixture frames I authored — is confirmed
by the live service. Not one needed changing, and nothing above the seam needed
editing at all:

| Confirmed live | Normalized |
|---|---|
| `response.created` | `AGENT_TURN_STARTED` |
| `response.output_audio.delta` | `AGENT_AUDIO` |
| `response.output_audio_transcript.delta` / `.done` | `AGENT_TRANSCRIPT` |
| `response.function_call_arguments.done` | `TOOL_CALL` |
| `response.done` (status `completed`) | `AGENT_TURN_COMPLETE` |
| `response.done` (status `cancelled`) | `INTERRUPTED` |
| `conversation.item.input_audio_transcription.completed` / `.delta` | `CALLER_TRANSCRIPT` |
| `error` | `ERROR` |

GA also emits `conversation.item.added/done`, `response.output_item.added/done`,
`response.content_part.added/done`, `rate_limits.updated` and
`input_audio_buffer.committed`. The harness has no meaning for any of them and
translates them to nothing, which is the correct behaviour for a chatty
protocol — a vendor adding a frame should not crash a consumer.

*The client-to-server session shape was wrong, and structurally.* The adapter was
written against the Realtime **Beta** API, which is switched off: sending
`OpenAI-Beta: realtime=v1` closes the socket immediately with
`beta_api_shape_disabled` and "The Realtime Beta API is no longer supported."
The GA shape differs in kind, not detail:

| Beta (dead) | GA (live) |
|---|---|
| `OpenAI-Beta: realtime=v1` header | header must be **absent** |
| `session.modalities` | `session.output_modalities` |
| `session.input_audio_format: "pcm16"` | `session.audio.input.format: {"type":"audio/pcm","rate":24000}` |
| `session.output_audio_format: "pcm16"` | `session.audio.output.format: {…}` |
| `session.voice` | `session.audio.output.voice` |
| `session.turn_detection` | `session.audio.input.turn_detection` |
| `session.input_audio_transcription` | `session.audio.input.transcription` |

**Honest verdict on protocol comparability:** the two vendors' *event streams*
really are interchangeable behind one normalization layer — that half of the
seam claim is now evidence rather than assertion. Their *session-configuration
surfaces* are not interchangeable, and are not even stable across one vendor's
own versions. That is an argument for the seam, not against it: all of the churn
was confined to one method, `session_update_frame`, and none of it reached the
measurement layer.

### Where OpenAI Realtime is better than Gemini Live

Not merely different — better, on three counts that mattered to this harness:

1. **It emits `response.created` before any audio.** `emits_turn_start` is True,
   so its turns decompose one stage further. Gemini Live sends no such frame, so
   everything from caller-stop to first audio is one opaque block there.
2. **Explicit turn boundaries are a first-class control path.**
   `input_audio_buffer.commit` + `response.create` is the documented way to
   drive it. On Gemini I had to *disable* automatic voice-activity detection
   because it cancelled the agent's in-flight tool calls at the end of every
   caller utterance.
3. **Function-call round trips were stable.** No analogue of Gemini's
   cancel-and-reissue behaviour appeared, so no de-duplication guard was needed
   on this side.

Where Gemini Live is better: it accepts 16 kHz caller audio (OpenAI GA wants
24 kHz), and its `serverContent.interrupted` is a clearer barge-in signal than
inferring an interruption from a `response.done` carrying status `cancelled`.

### Shared failure mode: spelled-out identifiers

Both stacks garble the thing a support line depends on most. Asked to read back
an account, Gemini heard `acct one zero four two` as **"ACTT1042"**; OpenAI heard
`p, r, i, y, a at northwindlabs dot io` as **"priyaa@northwindlabs.io"** and then
could not find the account at all. This is not a vendor difference — it is the
central difficulty of voice support, and it is invisible to any evaluation that
does not check the *arguments* the agent passed to its tools.

## Adding a third stack

The procedure that worked here, in the order it worked:

1. Probe the socket by hand before writing an adapter. Two of the three things
   that broke — the dead Beta shape and Gemini's `mediaChunks` rejection —
   present as connection failures, not as protocol errors, and cost far more to
   diagnose from inside a harness than from a 30-line script.
2. Write `translate()` from the docs and unit-test it against fixture frames.
   This part transferred intact for OpenAI, and it is the part the measurement
   layer depends on.
3. Expect `session_update_frame()` to be wrong. Confine every vendor assumption
   to it.
4. Flip `wire_verified = True` **only** after a real call completes end to end,
   and pin the observed event names in `tests/test_provider_seam.py` so a later
   edit cannot break them silently.

## The third adapter: `mock`

`MockVoiceProvider` is a small simulated realtime server on a virtual clock. It
implements the whole contract — streaming audio in chunks, requesting tools,
waiting for results, honouring barge-in — against a script.

It exists so the measurement layer can be tested against known truth: when the
script says "respond 620 ms after the caller stops, speak for 3.1 s, and yield
180 ms after being interrupted", the metrics have a right answer to check
against. It is also what lets the full pipeline run end to end with no API key
and no spend. Every record it produces is flagged `synthetic: true`, and the
report groups on that flag so a simulated run can never be shown as a real one.
