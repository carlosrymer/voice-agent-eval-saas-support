"""OpenAI Realtime adapter (GA `/v1/realtime` over WebSocket).

**Status: wire-verified.** Real frames round-trip against the live service:
session configuration, caller audio in, speech transcription, a function call
out, a tool result back in, and agent audio out.

## What the seam actually cost, honestly

The project claimed a second vendor would be "a translation table, not a
rewrite". Half of that held and half did not, and the split is the interesting
part.

**The server-to-client translation table was exactly right.** Every event name
in :func:`translate` — written from published docs months before a key existed,
and unit-tested only against fixture frames I authored — is confirmed by the
live service: ``response.created``, ``response.output_audio.delta``,
``response.output_audio_transcript.delta``/``.done``,
``response.function_call_arguments.done``, ``response.done`` with a status,
``conversation.item.input_audio_transcription.completed``, ``error``. Not one
needed changing. The measurement layer above the seam needed no edits at all.

**The client-to-server session shape was wrong, and not slightly.** The adapter
was written against the Realtime *Beta* API, which is now switched off: the
socket closes immediately with ``beta_api_shape_disabled`` and the message
"The Realtime Beta API is no longer supported." The GA API differs
structurally, not cosmetically:

* the ``OpenAI-Beta: realtime=v1`` header must be **absent**, not updated
* ``modalities`` became ``output_modalities``
* flat ``input_audio_format`` / ``output_audio_format`` strings became nested
  ``audio.input.format`` / ``audio.output.format`` objects
  (``{"type": "audio/pcm", "rate": 24000}``)
* ``voice``, ``turn_detection`` and ``input_audio_transcription`` all moved
  under ``audio.input`` / ``audio.output``

So the honest verdict on protocol comparability: **the event streams of these
two vendors really are interchangeable behind one normalization layer; their
session-configuration surfaces are not, and are not stable over time either.**
Everything vendor-specific in this file is confined to
:meth:`session_update_frame`, which is where that instability belongs.

## Where OpenAI Realtime is better than Gemini Live

Not just different — better, on three counts that matter to this harness:

* **It emits ``response.created`` before any audio**, so ``emits_turn_start`` is
  True and its turns decompose one stage further than Gemini's. Gemini gives one
  opaque block from caller-stop to first audio.
* **Turn boundaries are explicit by design.** ``input_audio_buffer.commit`` plus
  ``response.create`` is a first-class control path, not a workaround. On Gemini
  I had to disable automatic voice-activity detection because it cancelled the
  agent's own in-flight tool calls at the end of every caller utterance.
* **Function-call round trips were stable.** No analogue of Gemini's
  cancel-and-reissue behaviour appeared, so no de-duplication guard was needed.

## The mapping

| OpenAI Realtime server event | Normalized event |
|---|---|
| `session.created` | `SESSION_OPENED` |
| `input_audio_buffer.speech_started` | `CALLER_SPEECH_STARTED` |
| `input_audio_buffer.speech_stopped` | `CALLER_SPEECH_STOPPED` |
| `conversation.item.input_audio_transcription.completed` | `CALLER_TRANSCRIPT` (final) |
| `response.created` | `AGENT_TURN_STARTED` |
| `response.output_audio.delta` / `response.audio.delta` | `AGENT_AUDIO` |
| `response.output_audio_transcript.delta` / `.audio_transcript.delta` | `AGENT_TRANSCRIPT` |
| `response.output_audio_transcript.done` | `AGENT_TRANSCRIPT` (final) |
| `response.function_call_arguments.done` | `TOOL_CALL` |
| `response.done` | `AGENT_TURN_COMPLETE` |
| `error` | `ERROR` |

Both the current and legacy event names are accepted, because the audio delta
event was renamed and third-party examples disagree about which is live.

## The capability difference that changes a metric

Unlike Gemini Live, OpenAI Realtime **does** emit a distinct ``response.created``
before any audio. That means ``emits_turn_start`` is ``True`` here and ``False``
for Gemini, and the latency report will decompose an OpenAI turn one stage
further than a Gemini turn. This is exactly the situation the capability
descriptors exist for: the comparison stays honest because the extra stage
appears as a named stage for one provider and as unattributed time for the
other, rather than one provider silently appearing faster.

The other difference is turn-taking: OpenAI's server VAD can be disabled in
favour of explicit ``input_audio_buffer.commit``, so :meth:`commit_turn` is a
real operation here where it is a no-op for Gemini.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import AsyncIterator
from typing import Any

from voiceval.audio.pcm import PCM, resample
from voiceval.providers.base import (
    Clock,
    EventKind,
    ProviderCapabilities,
    ProviderUnavailable,
    ServerEvent,
    SessionConfig,
    TurnDetection,
    VoiceProvider,
    VoiceSession,
    WallClock,
    register_provider,
)

WS_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime-mini"
DEFAULT_VOICE = "alloy"
#: OpenAI rejects any voice outside this set -- and rejects the *entire*
#: session.update frame when one appears. See `_map_voice`.
OPENAI_VOICES = {
    "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse",
    "marin", "cedar",
}
#: Cross-vendor aliases, so one CallConfig drives either stack unchanged.
VOICE_ALIASES = {"puck": "alloy", "kore": "shimmer", "charon": "ash", "fenrir": "verse"}
#: OpenAI Realtime speaks 24 kHz PCM16 in both directions.
INPUT_RATE = 24000
OUTPUT_RATE = 24000
SEND_CHUNK_MS = 20.0
TRANSCRIPTION_MODEL = "whisper-1"

AUDIO_DELTA_TYPES = {"response.output_audio.delta", "response.audio.delta"}
TRANSCRIPT_DELTA_TYPES = {
    "response.output_audio_transcript.delta",
    "response.audio_transcript.delta",
}
TRANSCRIPT_DONE_TYPES = {
    "response.output_audio_transcript.done",
    "response.audio_transcript.done",
}


def translate(frame: dict[str, Any], t: float, seq: int) -> list[ServerEvent]:
    """Pure vendor-frame -> normalized-event translation.

    Kept free of sockets, tasks and state so it can be tested exhaustively
    without credentials. Unknown frame types translate to nothing, which is the
    correct behaviour for a chatty protocol: the harness should ignore events it
    has no meaning for rather than crash on the vendor adding one.
    """
    kind = frame.get("type", "")

    def ev(k: EventKind, **kw: Any) -> ServerEvent:
        return ServerEvent(kind=k, t=t, seq=seq, raw={"type": kind}, **kw)

    if kind == "session.created":
        return [ev(EventKind.SESSION_OPENED)]

    if kind == "input_audio_buffer.speech_started":
        return [ev(EventKind.CALLER_SPEECH_STARTED)]

    if kind == "input_audio_buffer.speech_stopped":
        return [ev(EventKind.CALLER_SPEECH_STOPPED)]

    if kind == "conversation.item.input_audio_transcription.completed":
        return [ev(EventKind.CALLER_TRANSCRIPT, text=frame.get("transcript") or "", is_final=True)]

    if kind == "conversation.item.input_audio_transcription.delta":
        return [ev(EventKind.CALLER_TRANSCRIPT, text=frame.get("delta") or "", is_final=False)]

    if kind == "response.created":
        return [ev(EventKind.AGENT_TURN_STARTED)]

    if kind in AUDIO_DELTA_TYPES:
        delta = frame.get("delta") or ""
        if not delta:
            return []
        return [ev(EventKind.AGENT_AUDIO, audio=PCM(base64.b64decode(delta), OUTPUT_RATE))]

    if kind in TRANSCRIPT_DELTA_TYPES:
        return [ev(EventKind.AGENT_TRANSCRIPT, text=frame.get("delta") or "", is_final=False)]

    if kind in TRANSCRIPT_DONE_TYPES:
        return [ev(EventKind.AGENT_TRANSCRIPT, text=frame.get("transcript") or "", is_final=True)]

    if kind == "response.function_call_arguments.done":
        args: dict[str, Any] = {}
        raw_args = frame.get("arguments")
        if isinstance(raw_args, str) and raw_args.strip():
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        return [
            ev(
                EventKind.TOOL_CALL,
                call_id=str(frame.get("call_id") or frame.get("item_id") or ""),
                tool_name=frame.get("name") or "",
                tool_args=args,
            )
        ]

    if kind == "response.done":
        # A response cancelled by barge-in also arrives as response.done, with
        # a cancelled status. Distinguishing them here is what keeps the
        # barge-in metric comparable with Gemini's explicit interrupt frame.
        status = ((frame.get("response") or {}).get("status")) or ""
        if status == "cancelled":
            return [ev(EventKind.INTERRUPTED, message="response cancelled")]
        return [ev(EventKind.AGENT_TURN_COMPLETE)]

    if kind == "error":
        err = frame.get("error") or {}
        return [ev(EventKind.ERROR, message=str(err.get("message") or err or "error"))]

    return []


class OpenAIRealtimeSession(VoiceSession):
    def __init__(self, provider: "OpenAIRealtimeProvider", ws, config: SessionConfig, clock: Clock):
        self.provider = provider
        self.ws = ws
        self.config = config
        self.clock = clock
        self.capabilities = provider.capabilities()
        self._queue: asyncio.Queue[ServerEvent] = asyncio.Queue()
        self._seq = 0
        self._closed = False
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        try:
            async for raw in self.ws:
                t = self.clock.now()
                try:
                    frame = json.loads(raw if isinstance(raw, str) else raw.decode())
                except Exception:
                    continue
                for ev in translate(frame, t, self._next_seq()):
                    self._queue.put_nowait(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - no wire to exercise
            self._seq += 1
            self._queue.put_nowait(
                ServerEvent(EventKind.ERROR, self.clock.now(), self._seq,
                            message=f"{type(exc).__name__}: {exc}")
            )

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def send_audio(self, pcm: PCM, *, ground_truth_text: str | None = None) -> None:
        audio = resample(pcm, INPUT_RATE)
        chunk_n = int(INPUT_RATE * SEND_CHUNK_MS / 1000.0) * 2
        for off in range(0, len(audio.data), chunk_n):
            chunk = audio.data[off : off + chunk_n]
            await self.ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }
                )
            )
            if self.provider.realtime_pacing:
                await asyncio.sleep(len(chunk) / 2 / INPUT_RATE)

    async def commit_turn(self) -> None:
        if self.config.turn_detection is TurnDetection.CLIENT_COMMIT:
            await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await self.ws.send(json.dumps({"type": "response.create"}))

    async def send_tool_result(self, call_id: str, name: str, payload: Any) -> None:
        await self.ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(payload, default=str),
                    },
                }
            )
        )
        await self.ws.send(json.dumps({"type": "response.create"}))

    async def cancel_response(self) -> None:
        await self.ws.send(json.dumps({"type": "response.cancel"}))

    async def events(self) -> AsyncIterator[ServerEvent]:
        while not self._closed:
            try:
                ev = await asyncio.wait_for(self._queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                return
            yield ev

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._recv_task.cancel()
        try:
            await self.ws.close()
        except Exception:
            pass


@register_provider
class OpenAIRealtimeProvider(VoiceProvider):
    name = "openai_realtime"
    #: Verified: session config, audio in, transcription, tool call, audio out.
    wire_verified = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        api_key: str | None = None,
        realtime_pacing: bool = True,
    ):
        self.model = model
        self.voice = voice
        # `None` means "look it up"; an explicit "" means "there is no key",
        # which is how tests assert the no-credentials path. `or` conflated the
        # two and silently used the ambient key once one existed.
        self.api_key = os.environ.get("OPENAI_API_KEY", "") if api_key is None else api_key
        self.realtime_pacing = realtime_pacing

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            server_turn_detection=True,
            server_barge_in=True,
            emits_interrupt_event=True,
            emits_caller_transcript=True,
            emits_agent_transcript=True,
            # The one behavioural difference from Gemini that changes a metric.
            emits_turn_start=True,
            input_sample_rate_hz=INPUT_RATE,
            output_sample_rate_hz=OUTPUT_RATE,
            input_mime="audio/pcm;rate=24000",
            output_mime="audio/pcm;rate=24000",
            native_tool_calling=True,
            notes=(
                "GA /v1/realtime, wire-verified. Explicit "
                "input_audio_buffer.commit + response.create turn boundaries, so "
                "end-of-turn latency excludes server-side endpointing, matching "
                "how Gemini Live is driven here. Emits an explicit "
                "response-start frame, so its latency decomposes one stage "
                "further than Gemini Live's."
            ),
        )

    def _map_voice(self, voice: str) -> str:
        """Coerce a voice name into one OpenAI accepts.

        Not cosmetic. Passing Gemini's default voice name straight through made
        OpenAI reject the whole `session.update` frame, and a rejected frame
        does not close the socket -- the session silently keeps its defaults.
        That left server voice-activity detection on and input transcription
        off, so the harness spent an entire run fighting auto-created responses
        and recording empty caller transcripts. One bad enum value degraded
        every downstream metric and nothing in the transcript said so.
        """
        v = (voice or "").strip()
        if v in OPENAI_VOICES:
            return v
        return VOICE_ALIASES.get(v.lower(), DEFAULT_VOICE)


    def session_update_frame(self, config: SessionConfig) -> dict[str, Any]:
        """The GA `session.update` frame, exposed so tests can assert its shape.

        This is the only place the vendor's configuration surface leaks, and it
        is the only part of the adapter the Beta-to-GA transition invalidated.
        """
        input_audio: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": INPUT_RATE},
            "turn_detection": (
                None
                if config.turn_detection is TurnDetection.CLIENT_COMMIT
                else {"type": "server_vad"}
            ),
        }
        if config.request_input_transcript:
            input_audio["transcription"] = {"model": TRANSCRIPTION_MODEL}

        session: dict[str, Any] = {
            "type": "realtime",
            "instructions": config.system_instruction,
            "output_modalities": ["audio"],
            "audio": {
                "input": input_audio,
                "output": {
                    "format": {"type": "audio/pcm", "rate": OUTPUT_RATE},
                    "voice": self._map_voice(config.voice or self.voice),
                },
            },
        }
        if config.tools:
            session["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in config.tools
            ]
            session["tool_choice"] = "auto"
        if config.max_output_tokens is not None:
            session["max_output_tokens"] = config.max_output_tokens
        return {"type": "session.update", "session": session}

    async def connect(self, config: SessionConfig, clock: Clock | None = None) -> VoiceSession:
        if not self.api_key:
            raise ProviderUnavailable("OPENAI_API_KEY is not set")
        import websockets

        clock = clock or WallClock()
        try:
            ws = await websockets.connect(
                f"{WS_URL}?model={self.model}",
                # No OpenAI-Beta header. Sending it selects the discontinued
                # Beta shape and the server closes the socket immediately with
                # `beta_api_shape_disabled`.
                additional_headers={"Authorization": f"Bearer {self.api_key}"},
                max_size=None,
                open_timeout=30,
            )
        except Exception as exc:
            raise ProviderUnavailable(f"could not open Realtime WebSocket: {exc}") from exc

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
        except asyncio.TimeoutError as exc:
            await ws.close()
            raise ProviderUnavailable("no session.created from Realtime API") from exc
        first = json.loads(raw if isinstance(raw, str) else raw.decode())
        if first.get("type") == "error":
            await ws.close()
            raise ProviderUnavailable(f"Realtime rejected the session: {json.dumps(first)[:300]}")

        await ws.send(json.dumps(self.session_update_frame(config)))
        # Confirm the configuration was accepted. A rejected session.update
        # leaves the socket open on default settings, so failing loudly here is
        # the difference between a wrong measurement and no measurement.
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
        except asyncio.TimeoutError as exc:
            await ws.close()
            raise ProviderUnavailable("no reply to session.update") from exc
        ack = json.loads(raw if isinstance(raw, str) else raw.decode())
        if ack.get("type") == "error":
            await ws.close()
            raise ProviderUnavailable(
                f"session.update rejected: {json.dumps(ack.get('error') or ack)[:300]}"
            )
        return OpenAIRealtimeSession(self, ws, config, clock)
