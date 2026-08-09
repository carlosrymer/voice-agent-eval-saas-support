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

### `openai_realtime` — translation layer only, wire unverified

There is no OpenAI key in this environment, so `wire_verified = False`,
`connect()` refuses without a key, and the capability notes say `UNVERIFIED` so
it propagates into any report that includes the provider.

What *is* tested is `translate(frame, t, seq)`, a pure function from a vendor
frame to normalized events, driven in `tests/test_provider_seam.py` by
hand-written OpenAI-shaped frames. That is the part of the claim — "a second
vendor is a translation table" — that can be checked without credentials.

| OpenAI Realtime server event | Normalized |
|---|---|
| `session.created` | `SESSION_OPENED` |
| `input_audio_buffer.speech_started` / `.speech_stopped` | `CALLER_SPEECH_STARTED` / `STOPPED` |
| `conversation.item.input_audio_transcription.completed` | `CALLER_TRANSCRIPT` (final) |
| `response.created` | `AGENT_TURN_STARTED` |
| `response.output_audio.delta` *or* `response.audio.delta` | `AGENT_AUDIO` |
| `response.output_audio_transcript.delta` / `.done` | `AGENT_TRANSCRIPT` |
| `response.function_call_arguments.done` | `TOOL_CALL` |
| `response.done` status `completed` | `AGENT_TURN_COMPLETE` |
| `response.done` status `cancelled` | `INTERRUPTED` |
| `error` | `ERROR` |

Two mappings there are load-bearing and are covered by their own tests. A
cancelled response must normalize to `INTERRUPTED`, not to a completed turn, or
barge-in silently reads zero for this provider while reading correctly for
Gemini. And both the current and legacy audio-delta event names are accepted,
because the event was renamed and examples in the wild disagree about which is
live.

The one capability difference that changes a metric: OpenAI Realtime *does* emit
`response.created` before any audio, so `emits_turn_start=True` and its turns
decompose one stage further than Gemini's. That is the situation the capability
descriptors exist for — the extra stage appears as a named stage for one
provider and as residual for the other, rather than one vendor simply looking
faster.

## To finish the OpenAI arm when a key exists

1. `export OPENAI_API_KEY=…`
2. `uv run python -m voiceval.run_experiment --provider openai_realtime --trials 1`
3. Fix whatever the wire disagrees with — expect the disagreements to be in
   `translate()` and `session_update_frame()`, which is where every
   vendor-specific assumption lives.
4. Flip `wire_verified = True` **only** after a real call completes, and add the
   observed frame shapes to `tests/test_provider_seam.py` so the next change
   cannot break them silently.

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
