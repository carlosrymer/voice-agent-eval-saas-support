"""Text-to-speech for the simulated caller.

The caller's voice is a cost centre and a confound, so both get handled here.

*Cost*: every synthesised utterance is cached on disk keyed by the exact text,
voice and model. Re-running the experiment after a scoring change re-uses the
audio instead of re-billing for it, and -- more usefully -- the caller audio
stays byte-identical across runs, so a change in a measured number is a change
in the agent rather than in what it was played.

*Confound*: the caller is synthesised by the same vendor as the agent under
test. A Gemini agent hearing a Gemini voice is an easier listening task than a
real human caller on a phone line, and it flatters the word-error-rate figure.
That is stated in the README; the interface here is provider-neutral so a
different TTS vendor can be dropped in when one is available.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from voiceval.audio import fixtures as fx
from voiceval.audio.pcm import PCM, read_wav, write_wav

GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"

#: Tried in order. Gemini meters `generate_requests_per_model_per_day` **per
#: model** (100/day on this key), so a second and third TTS model is not
#: redundancy for reliability -- it is three separate daily budgets. Exhausting
#: the first one mid-experiment is exactly what happened on the first full run.
TTS_MODEL_CHAIN = (
    "gemini-2.5-flash-preview-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-pro-preview-tts",
)
GEMINI_TTS_RATE = 24000
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class TTSQuotaExhausted(RuntimeError):
    """Every TTS model in the chain returned 429.

    A distinct type because the orchestrator must treat this as *the harness
    ran out of budget*, not as the agent failing the task. Scoring a
    quota-truncated call as an agent failure would manufacture exactly the
    voice-channel degradation this project set out to measure.
    """


class TTSBackend(ABC):
    name: str = "tts"
    #: Sample rate of the audio this backend returns.
    sample_rate_hz: int = 24000

    @abstractmethod
    def synthesize(self, text: str, voice: str) -> PCM: ...

    #: Tokens billed so far, for the spend report.
    usage: dict[str, int]


class FixtureTTS(TTSBackend):
    """Offline stand-in. Produces speech-shaped audio of a plausible length."""

    name = "fixture"

    def __init__(self, sample_rate_hz: int = 16000, words_per_second: float = 2.6):
        self.sample_rate_hz = sample_rate_hz
        self.words_per_second = words_per_second
        self.usage = {}

    def synthesize(self, text: str, voice: str) -> PCM:
        n_words = max(1, len(text.split()))
        duration = max(0.35, n_words / self.words_per_second)
        # Seeded on the text so the same line always renders identically.
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        return fx.speech_like(duration, f0_hz=185.0, rate=self.sample_rate_hz, seed=seed)


class GeminiTTS(TTSBackend):
    """Gemini TTS over ``generateContent`` with ``responseModalities: ["AUDIO"]``.

    Returns raw little-endian 16-bit PCM at 24 kHz with **no container** -- the
    response mime is ``audio/l16; rate=24000; channels=1`` and the bytes start
    at sample zero. Handing that to anything expecting a WAV produces four
    milliseconds of noise where the header would have been, so it is wrapped
    into :class:`PCM` here and written out with a real header only at the edge.
    """

    name = "gemini_tts"
    sample_rate_hz = GEMINI_TTS_RATE

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        timeout_s: float = 120.0,
        models: tuple[str, ...] = TTS_MODEL_CHAIN,
    ):
        self.models = (model,) if model else tuple(models)
        self.model = self.models[0]
        self.exhausted: set[str] = set()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.cache_dir = Path(cache_dir) if cache_dir else Path("artifacts/tts_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.usage: dict[str, int] = {}
        self.n_calls = 0
        self.n_cache_hits = 0

    def _key(self, text: str, voice: str, model: str | None = None) -> str:
        return hashlib.sha256(
            f"{model or self.model}|{voice}|{text}".encode()
        ).hexdigest()[:24]

    def synthesize(self, text: str, voice: str) -> PCM:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        # The cache is keyed on the *text*, not the model, so an utterance
        # already rendered costs nothing and keeps the caller's voice
        # byte-identical across re-runs.
        for m in self.models:
            cached = self.cache_dir / f"{self._key(text, voice, m)}.wav"
            if cached.exists():
                self.n_cache_hits += 1
                return read_wav(str(cached))

        errors: list[str] = []
        for m in self.models:
            if m in self.exhausted:
                continue
            try:
                pcm = self._synthesize_with(m, text, voice)
            except _QuotaError as exc:
                self.exhausted.add(m)
                errors.append(f"{m}: {exc}")
                continue
            except Exception as exc:
                errors.append(f"{m}: {exc}")
                continue
            write_wav(str(self.cache_dir / f"{self._key(text, voice, m)}.wav"), pcm)
            self.model = m
            return pcm
        raise TTSQuotaExhausted("; ".join(errors) or "no TTS model available")

    def _synthesize_with(self, model: str, text: str, voice: str) -> PCM:
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
                },
            },
        }
        url = f"{API_BASE}/{model}:generateContent?key={self.api_key}"
        last: Exception | None = None
        for attempt in range(3):
            r = httpx.post(url, json=payload, timeout=self.timeout_s)
            if r.status_code == 429:
                # Daily per-model quota. Retrying will not help today.
                raise _QuotaError(r.text[:160])
            if r.status_code >= 500:
                last = RuntimeError(f"HTTP {r.status_code}")
                import time

                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            break
        else:
            raise RuntimeError(f"{model} failed: {last}")

        self.n_calls += 1
        meta = data.get("usageMetadata") or {}
        for k, v in meta.items():
            if isinstance(v, int):
                self.usage[k] = self.usage.get(k, 0) + v
        for d in meta.get("candidatesTokensDetails", []) or []:
            if isinstance(d.get("tokenCount"), int):
                kk = f"out_{str(d.get('modality', 'unknown')).lower()}"
                self.usage[kk] = self.usage.get(kk, 0) + d["tokenCount"]
        self.usage["calls_" + model] = self.usage.get("calls_" + model, 0) + 1

        try:
            part = data["candidates"][0]["content"]["parts"][0]["inlineData"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"unexpected TTS response: {json.dumps(data)[:300]}") from exc
        return PCM(base64.b64decode(part["data"]), _rate_from_mime(part.get("mimeType", "")))


class _QuotaError(RuntimeError):
    pass


def _rate_from_mime(mime: str) -> int:
    for bit in mime.replace(";", " ").split():
        if bit.startswith("rate="):
            try:
                return int(bit.split("=", 1)[1])
            except ValueError:
                pass
    return GEMINI_TTS_RATE


def get_tts(name: str, **kwargs) -> TTSBackend:
    if name in ("gemini", "gemini_tts"):
        return GeminiTTS(**kwargs)
    if name == "fixture":
        return FixtureTTS(**{k: v for k, v in kwargs.items() if k in {"sample_rate_hz"}})
    raise KeyError(f"unknown TTS backend {name!r}")
