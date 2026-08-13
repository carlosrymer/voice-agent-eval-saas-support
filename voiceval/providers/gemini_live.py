"""Gemini Live API adapter (``bidiGenerateContent`` over WebSocket).

Wire facts, all verified against the live endpoint rather than taken from docs:

* URL is ``wss://generativelanguage.googleapis.com/ws/
  google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent``
  with the API key as a query parameter.
* The first client frame is ``setup``; the server replies ``setupComplete``.
* Caller audio goes up as ``realtimeInput.audio`` with mime
  ``audio/pcm;rate=16000``. (``realtimeInput.mediaChunks``, which most
  third-party examples still show, is rejected outright: the socket closes with
  1007 and a deprecation message rather than an error frame.) Agent audio comes
  back as
  ``serverContent.modelTurn.parts[].inlineData`` at ``audio/pcm;rate=24000``.
* Tool calls arrive as ``toolCall.functionCalls[]`` and are answered with
  ``toolResponse.functionResponses[]`` keyed by the same id.
* Barge-in shows up as ``serverContent.interrupted``.

Two design points worth stating.

**Receiving runs in its own task.** The socket is drained continuously into a
queue by :meth:`_recv_loop`, and every frame is timestamped the moment it is
read. That is what makes barge-in possible at all -- the orchestrator can push
caller audio while the agent is still talking, because nothing is blocked on a
synchronous read. It is also what makes the latency numbers mean anything: the
timestamp is taken before any parsing, so it measures the network and the
model, not this adapter's JSON handling.

**Turn boundaries are signalled explicitly by default.** Gemini Live's
automatic VAD proved unusable for this harness: at the end of every caller
utterance the server re-detected activity and cancelled the agent's in-flight
tool call about 190 ms after issuing it, so no task involving a tool could ever
complete. The fix is ``automaticActivityDetection.disabled`` plus explicit
``activityStart`` / ``activityEnd`` frames. This costs fidelity and the cost is
stated everywhere the latency is reported: **end-of-turn latency measured this
way excludes the server's own endpointing delay**, which a production deployment
on automatic VAD would add on top (typically a few hundred milliseconds). It
buys determinism -- turn boundaries and barge-in points are now exactly where
the harness put them, which is what makes the barge-in measurements comparable
between runs. Pass ``manual_activity=False`` to measure the other way.

**Caller audio is transmitted as a burst, not streamed at speaking pace.**
Streaming it in real time let the server begin generating part-way through the
caller's utterance and then cancel that generation when the turn closed,
discarding the agent's in-flight tool call with it. The orchestrator instead
spends the utterance's duration in real time and then sends it in one go inside
an explicit activity window; see :func:`voiceval.orchestrator.run_call` for the
timing and for what it costs. ``realtime_pacing=True`` restores streaming and is
recorded in the run manifest, because runs made the two ways are not
comparable.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import websockets

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

WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

INPUT_RATE = 16000
OUTPUT_RATE = 24000
SEND_CHUNK_MS = 20.0

#: How long `events()` waits with nothing arriving before handing control back
#: to the orchestrator. Not a timeout on the call -- just the point at which
#: "the provider is waiting for me" becomes the better assumption.
IDLE_YIELD_S = 0.25

DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_VOICE = "Puck"


class GeminiLiveSession(VoiceSession):
    def __init__(self, provider: "GeminiLiveProvider", ws, config: SessionConfig, clock: Clock):
        self.provider = provider
        self.ws = ws
        self.config = config
        self.clock = clock
        self.capabilities = provider.capabilities()
        self._queue: asyncio.Queue[ServerEvent] = asyncio.Queue()
        self._seq = 0
        self._closed = False
        self._turn_open = False
        self._recv_task = asyncio.create_task(self._recv_loop())
        #: Partial transcripts, accumulated until a turn boundary makes them final.
        self._caller_buf: list[str] = []
        self._agent_buf: list[str] = []
        self._activity_open = False
        self.usage: dict[str, int] = {}

    # -- plumbing ----------------------------------------------------------
    def _put(self, kind: EventKind, **kw: Any) -> None:
        self._seq += 1
        self._queue.put_nowait(ServerEvent(kind=kind, t=self.clock.now(), seq=self._seq, **kw))

    async def _recv_loop(self) -> None:
        try:
            async for raw in self.ws:
                # Stamp first, parse second.
                t = self.clock.now()
                try:
                    msg = json.loads(raw if isinstance(raw, str) else raw.decode())
                except Exception:
                    continue
                if os.environ.get("VOICEVAL_WIRE_DEBUG"):
                    sc = msg.get("serverContent") or {}
                    flags = [k for k in ("turnComplete", "generationComplete",
                                         "interrupted") if sc.get(k)]
                    has_audio = any(
                        "inlineData" in p for p in
                        ((sc.get("modelTurn") or {}).get("parts") or [])
                    )
                    print(f"[wire {t:7.2f}] {list(msg.keys())} {flags}"
                          f"{' +audio' if has_audio else ''}", flush=True)
                self._handle(msg, t)
        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed:
            self._put(EventKind.SESSION_CLOSED)
        except Exception as exc:  # pragma: no cover - defensive
            self._put(EventKind.ERROR, message=f"{type(exc).__name__}: {exc}")

    def _emit_at(self, kind: EventKind, t: float, **kw: Any) -> None:
        self._seq += 1
        self._queue.put_nowait(ServerEvent(kind=kind, t=t, seq=self._seq, **kw))

    def _handle(self, msg: dict[str, Any], t: float) -> None:
        if "setupComplete" in msg:
            self._emit_at(EventKind.SESSION_OPENED, t)
            return

        if "toolCall" in msg:
            for fc in msg["toolCall"].get("functionCalls", []) or []:
                self._emit_at(
                    EventKind.TOOL_CALL,
                    t,
                    call_id=str(fc.get("id") or fc.get("name")),
                    tool_name=fc.get("name", ""),
                    tool_args=fc.get("args") or {},
                    raw={"toolCall": fc},
                )
            return

        if "toolCallCancellation" in msg:
            ids = msg["toolCallCancellation"].get("ids", []) or []
            self._emit_at(
                EventKind.INTERRUPTED, t, message=f"tool call cancelled: {ids}",
                raw={"toolCallCancellation": msg["toolCallCancellation"]},
            )
            return

        if "usageMetadata" in msg:
            self._merge_usage(msg["usageMetadata"])

        sc = msg.get("serverContent")
        if not sc:
            return

        if it := sc.get("inputTranscription"):
            if text := it.get("text"):
                self._caller_buf.append(text)
                self._emit_at(EventKind.CALLER_TRANSCRIPT, t, text=text, is_final=False)

        if ot := sc.get("outputTranscription"):
            if text := ot.get("text"):
                self._agent_buf.append(text)
                self._emit_at(EventKind.AGENT_TRANSCRIPT, t, text=text, is_final=False)

        parts = ((sc.get("modelTurn") or {}).get("parts")) or []
        for part in parts:
            inline = part.get("inlineData")
            if inline and str(inline.get("mimeType", "")).startswith("audio/"):
                if not self._turn_open:
                    self._turn_open = True
                    # The first audio byte is also the first observable sign of
                    # this response for providers that send no separate start
                    # frame. Emitting a turn-start here would invent a marker,
                    # so it is emitted only when the API gives us one.
                data = base64.b64decode(inline["data"])
                self._emit_at(EventKind.AGENT_AUDIO, t, audio=PCM(data, OUTPUT_RATE))
            elif part.get("text"):
                self._agent_buf.append(part["text"])
                self._emit_at(EventKind.AGENT_TRANSCRIPT, t, text=part["text"], is_final=False)

        if sc.get("generationComplete") and not self._turn_open:
            self._emit_at(EventKind.AGENT_TURN_STARTED, t)

        if sc.get("interrupted"):
            self._finalise_transcripts(t)
            self._turn_open = False
            self._emit_at(EventKind.INTERRUPTED, t, message="server reported interruption")
            return

        if sc.get("turnComplete"):
            self._finalise_transcripts(t)
            self._turn_open = False
            self._emit_at(EventKind.AGENT_TURN_COMPLETE, t)

    def _finalise_transcripts(self, t: float) -> None:
        if self._caller_buf:
            self._emit_at(
                EventKind.CALLER_TRANSCRIPT, t, text="".join(self._caller_buf).strip(),
                is_final=True,
            )
            self._caller_buf = []
        if self._agent_buf:
            self._emit_at(
                EventKind.AGENT_TRANSCRIPT, t, text="".join(self._agent_buf).strip(),
                is_final=True,
            )
            self._agent_buf = []

    def _merge_usage(self, usage: dict[str, Any]) -> None:
        for key in ("promptTokenCount", "responseTokenCount", "totalTokenCount"):
            if isinstance(usage.get(key), int):
                self.usage[key] = self.usage.get(key, 0) + usage[key]
        for bucket in ("promptTokensDetails", "responseTokensDetails"):
            for d in usage.get(bucket, []) or []:
                mod = str(d.get("modality", "UNKNOWN")).lower()
                n = d.get("tokenCount")
                if isinstance(n, int):
                    k = f"{bucket.replace('TokensDetails', '')}_{mod}"
                    self.usage[k] = self.usage.get(k, 0) + n

    # -- VoiceSession ------------------------------------------------------
    async def send_audio(self, pcm: PCM, *, ground_truth_text: str | None = None) -> None:
        if self._closed:
            raise RuntimeError("session is closed")
        audio = resample(pcm, INPUT_RATE)
        chunk_n = int(INPUT_RATE * SEND_CHUNK_MS / 1000.0) * 2
        pacing = self.provider.realtime_pacing
        if self.provider.manual_activity and not self._activity_open:
            await self.ws.send(json.dumps({"realtimeInput": {"activityStart": {}}}))
            self._activity_open = True
        for off in range(0, len(audio.data), chunk_n):
            chunk = audio.data[off : off + chunk_n]
            await self.ws.send(
                json.dumps(
                    {
                        "realtimeInput": {
                            "audio": {
                                "mimeType": f"audio/pcm;rate={INPUT_RATE}",
                                "data": base64.b64encode(chunk).decode(),
                            }
                        }
                    }
                )
            )
            if pacing:
                await asyncio.sleep(len(chunk) / 2 / INPUT_RATE)

    async def commit_turn(self) -> None:
        """Close the caller's activity window.

        In manual mode this is what tells the server the caller has finished, so
        it is the moment end-of-turn latency is measured from. In automatic-VAD
        mode it is a no-op and the server finds the boundary itself.
        """
        if self.provider.manual_activity and self._activity_open:
            await self.ws.send(json.dumps({"realtimeInput": {"activityEnd": {}}}))
            self._activity_open = False

    async def send_tool_result(self, call_id: str, name: str, payload: Any) -> None:
        if self._closed:
            raise RuntimeError("session is closed")
        if os.environ.get("VOICEVAL_WIRE_DEBUG"):
            print(f"[wire {self.clock.now():7.2f}] -> toolResponse id={call_id} {name}",
                  flush=True)
        response = payload if isinstance(payload, dict) else {"result": payload}
        await self.ws.send(
            json.dumps(
                {
                    "toolResponse": {
                        "functionResponses": [
                            {"id": call_id, "name": name, "response": response}
                        ]
                    }
                }
            )
        )

    async def cancel_response(self) -> None:
        """Gemini Live cancels its own output when it hears the caller.

        There is no client-side cancel frame in this API, so the honest
        implementation is to do nothing and let the capability flag tell the
        report that barge-in here is server-driven.
        """
        return

    async def events(self) -> AsyncIterator[ServerEvent]:
        while not self._closed:
            try:
                ev = await asyncio.wait_for(self._queue.get(), timeout=IDLE_YIELD_S)
            except asyncio.TimeoutError:
                return
            yield ev
            if ev.kind == EventKind.SESSION_CLOSED:
                return

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
class GeminiLiveProvider(VoiceProvider):
    name = "gemini_live"
    wire_verified = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        api_key: str | None = None,
        realtime_pacing: bool = False,
        manual_activity: bool = True,
    ):
        self.model = model
        self.voice = voice
        # `None` means "look it up"; an explicit "" means "there is no key",
        # which is how tests assert the no-credentials path. `or` conflated the
        # two and silently used the ambient key once one existed.
        self.api_key = os.environ.get("GEMINI_API_KEY", "") if api_key is None else api_key
        self.realtime_pacing = realtime_pacing
        #: Signal turn boundaries explicitly instead of letting the server's
        #: VAD find them. See the module docstring for why this is the default.
        self.manual_activity = manual_activity

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            server_turn_detection=not self.manual_activity,
            server_barge_in=True,
            emits_interrupt_event=True,
            emits_caller_transcript=True,
            emits_agent_transcript=True,
            # The API sends no distinct "response started" frame before audio,
            # so inference and speech synthesis cannot be separated for this
            # provider. The latency report says so rather than guessing.
            emits_turn_start=False,
            input_sample_rate_hz=INPUT_RATE,
            output_sample_rate_hz=OUTPUT_RATE,
            input_mime=f"audio/pcm;rate={INPUT_RATE}",
            output_mime=f"audio/pcm;rate={OUTPUT_RATE}",
            native_tool_calling=True,
            notes=(
                (
                    "Explicit activityStart/activityEnd turn signalling; "
                    "end-of-turn latency therefore EXCLUDES server-side "
                    "endpointing delay."
                    if self.manual_activity
                    else "Server-side VAD; end-of-turn latency includes endpointing."
                )
                + " Server-driven barge-in. No turn-start frame, so inference and "
                "TTS are not separable. Real-time input pacing "
                f"{'on' if self.realtime_pacing else 'OFF (latency not comparable)'}."
            ),
        )

    def _setup_frame(self, config: SessionConfig) -> dict[str, Any]:
        setup: dict[str, Any] = {
            "model": f"models/{self.model}",
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": config.voice or self.voice}
                    }
                },
            },
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        }
        if config.temperature is not None:
            setup["generationConfig"]["temperature"] = config.temperature
        if config.system_instruction:
            setup["systemInstruction"] = {"parts": [{"text": config.system_instruction}]}
        if config.tools:
            setup["tools"] = [
                {
                    "functionDeclarations": [
                        {"name": t.name, "description": t.description, "parameters": t.parameters}
                        for t in config.tools
                    ]
                }
            ]
        if self.manual_activity or config.turn_detection is TurnDetection.CLIENT_COMMIT:
            setup["realtimeInputConfig"] = {"automaticActivityDetection": {"disabled": True}}
        return setup

    async def connect(self, config: SessionConfig, clock: Clock | None = None) -> VoiceSession:
        if not self.api_key:
            raise ProviderUnavailable("GEMINI_API_KEY is not set")
        clock = clock or WallClock()
        try:
            ws = await websockets.connect(
                f"{WS_URL}?key={self.api_key}", max_size=None, open_timeout=30
            )
        except Exception as exc:
            raise ProviderUnavailable(f"could not open Live WebSocket: {exc}") from exc

        await ws.send(json.dumps({"setup": self._setup_frame(config)}))
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
        except asyncio.TimeoutError as exc:
            await ws.close()
            raise ProviderUnavailable("no setupComplete from Live API") from exc
        msg = json.loads(raw if isinstance(raw, str) else raw.decode())
        if "setupComplete" not in msg:
            await ws.close()
            raise ProviderUnavailable(f"Live setup rejected: {json.dumps(msg)[:400]}")
        return GeminiLiveSession(self, ws, config, clock)
