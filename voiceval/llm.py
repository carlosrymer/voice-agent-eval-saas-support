"""A small Gemini ``generateContent`` client for the non-realtime calls.

Three things in this project are ordinary request/response model calls rather
than realtime sessions: the caller simulator's brain, the transcript-only judge
and the audio-native judge. They share this client so that token accounting is
in one place and the spend report is a measurement rather than an estimate.

It is deliberately not a framework. Retries are bounded and only cover 429 and
5xx; a 400 is a bug in my request and retrying it just spends money slower.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_TEXT_MODEL = "gemini-3.6-flash"


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    by_modality: dict[str, int] = field(default_factory=dict)
    by_model: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, meta: dict[str, Any]) -> None:
        self.calls += 1
        self.prompt_tokens += int(meta.get("promptTokenCount") or 0)
        self.output_tokens += int(meta.get("candidatesTokenCount") or 0)
        self.total_tokens += int(meta.get("totalTokenCount") or 0)
        self.by_model[model] = self.by_model.get(model, 0) + int(meta.get("totalTokenCount") or 0)
        for bucket in ("promptTokensDetails", "candidatesTokensDetails"):
            for d in meta.get(bucket) or []:
                n = d.get("tokenCount")
                if isinstance(n, int):
                    k = f"{'in' if bucket.startswith('prompt') else 'out'}_{str(d.get('modality','unknown')).lower()}"
                    self.by_modality[k] = self.by_modality.get(k, 0) + n

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "by_modality": dict(sorted(self.by_modality.items())),
            "by_model": dict(sorted(self.by_model.items())),
        }


@dataclass
class LLMResponse:
    text: str
    function_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class GeminiClient:
    def __init__(self, api_key: str | None = None, timeout_s: float = 180.0):
        # `None` means "look it up"; an explicit "" means "there is no key",
        # which is how tests assert the no-credentials path. `or` conflated the
        # two and silently used the ambient key once one existed.
        self.api_key = os.environ.get("GEMINI_API_KEY", "") if api_key is None else api_key
        self.timeout_s = timeout_s
        self.usage = Usage()

    def generate(
        self,
        model: str,
        contents: list[dict[str, Any]],
        *,
        system_instruction: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_mime_type: str | None = None,
        max_attempts: int = 4,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        payload: dict[str, Any] = {"contents": contents}
        cfg: dict[str, Any] = {}
        if temperature is not None:
            cfg["temperature"] = temperature
        if max_output_tokens is not None:
            cfg["maxOutputTokens"] = max_output_tokens
        if response_mime_type:
            cfg["responseMimeType"] = response_mime_type
        if cfg:
            payload["generationConfig"] = cfg
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": tools}]

        url = f"{API_BASE}/{model}:generateContent?key={self.api_key}"
        last: Exception | None = None
        for attempt in range(max_attempts):
            try:
                r = httpx.post(url, json=payload, timeout=self.timeout_s)
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code >= 400:
                    # A 4xx that is not rate limiting is my mistake; surface it.
                    raise ValueError(f"HTTP {r.status_code}: {r.text[:400]}")
                data = r.json()
                break
            except ValueError:
                raise
            except Exception as exc:
                last = exc
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"generateContent failed: {last}") from exc
                time.sleep(2.0 * (attempt + 1))

        self.usage.add(model, data.get("usageMetadata") or {})
        text_bits: list[str] = []
        calls: list[dict[str, Any]] = []
        for cand in data.get("candidates", []) or []:
            for part in (cand.get("content") or {}).get("parts", []) or []:
                if "text" in part and part["text"]:
                    text_bits.append(part["text"])
                if "functionCall" in part:
                    fc = part["functionCall"]
                    calls.append({"name": fc.get("name", ""), "args": fc.get("args") or {}})
        return LLMResponse("".join(text_bits).strip(), calls, data)


OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_OPENAI_TEXT_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_AUDIO_MODEL = "gpt-audio"


class OpenAIClient:
    """Same `generate` contract as :class:`GeminiClient`, different vendor.

    It exists so the judge code is written once and the *vendor* becomes a
    parameter. That is what makes the cross-vendor experiment possible: the
    headline modality result was originally measured with a Gemini judge on
    Gemini calls, which is same-family self-evaluation. Holding the rubric, the
    prompt and the call fixed while swapping only the judge's vendor is the
    control that breaks that confound.

    Gemini-shaped `contents` are translated here rather than at the call site,
    so `judge_call` does not branch on vendor.
    """

    name = "openai"

    def __init__(self, api_key: str | None = None, timeout_s: float = 240.0):
        self.api_key = os.environ.get("OPENAI_API_KEY", "") if api_key is None else api_key
        self.timeout_s = timeout_s
        self.usage = Usage()

    @staticmethod
    def _translate(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        for c in contents:
            role = "assistant" if c.get("role") == "model" else "user"
            parts: list[dict[str, Any]] = []
            for p in c.get("parts", []):
                if "text" in p:
                    parts.append({"type": "text", "text": p["text"]})
                elif "inlineData" in p:
                    parts.append(
                        {
                            "type": "input_audio",
                            "input_audio": {"data": p["inlineData"]["data"], "format": "wav"},
                        }
                    )
            msgs.append({"role": role, "content": parts})
        return msgs

    def generate(
        self,
        model: str,
        contents: list[dict[str, Any]],
        *,
        system_instruction: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_mime_type: str | None = None,
        max_attempts: int = 4,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        messages = self._translate(contents)
        if system_instruction:
            messages.insert(0, {"role": "system", "content": system_instruction})
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None and not model.startswith("gpt-audio"):
            # The audio models reject an explicit temperature.
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_completion_tokens"] = max_output_tokens
        if response_mime_type == "application/json" and not model.startswith("gpt-audio"):
            # The audio models reject `response_format`, so the audio judge has
            # only the prompt's "JSON only" instruction holding it to shape.
            # That is exactly the condition `salvage_scores` was written for,
            # and it is why the audio arm is not disadvantaged by it.
            payload["response_format"] = {"type": "json_object"}

        last: Exception | None = None
        for attempt in range(max_attempts):
            try:
                r = httpx.post(
                    OPENAI_CHAT_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout_s,
                )
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code >= 400:
                    raise ValueError(f"HTTP {r.status_code}: {r.text[:400]}")
                data = r.json()
                break
            except ValueError:
                raise
            except Exception as exc:
                last = exc
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"chat.completions failed: {last}") from exc
                time.sleep(2.0 * (attempt + 1))

        u = data.get("usage") or {}
        self.usage.add(
            model,
            {
                "promptTokenCount": u.get("prompt_tokens"),
                "candidatesTokenCount": u.get("completion_tokens"),
                "totalTokenCount": u.get("total_tokens"),
                "promptTokensDetails": [
                    {"modality": "AUDIO",
                     "tokenCount": (u.get("prompt_tokens_details") or {}).get("audio_tokens")},
                    {"modality": "TEXT",
                     "tokenCount": (u.get("prompt_tokens_details") or {}).get("text_tokens")},
                ],
            },
        )
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        return LLMResponse(str(msg.get("content") or "").strip(), [], data)


def get_client(vendor: str):
    if vendor in ("gemini", "google"):
        return GeminiClient()
    if vendor in ("openai", "oai"):
        return OpenAIClient()
    raise KeyError(f"unknown judge vendor {vendor!r}")


def audio_part(wav_bytes: bytes, mime: str = "audio/wav") -> dict[str, Any]:
    return {"inlineData": {"mimeType": mime, "data": base64.b64encode(wav_bytes).decode()}}


def text_part(text: str) -> dict[str, Any]:
    return {"text": text}


def user_content(parts: list[dict[str, Any]]) -> dict[str, Any]:
    return {"role": "user", "parts": parts}


def model_content(parts: list[dict[str, Any]]) -> dict[str, Any]:
    return {"role": "model", "parts": parts}


def strip_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    s = s.strip()
    if s.startswith("json"):
        s = s[4:].strip()
    return s


def parse_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Judges are asked for JSON and mostly comply, but a fenced block or a
    sentence of preamble is common. Failing to parse must be visible rather than
    silently scored as zero, so this raises when it truly cannot.
    """
    s = strip_fences(text)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            return json.loads(s[start : end + 1])
        raise


#: `"criterion": { ... "score": <number|null> ... }` — tolerant of whatever the
#: rationale contains, including the unescaped quotes that break strict parsing.
_SCORE_RE = re.compile(
    r'"(?P<id>[a-z_]+)"\s*:\s*\{[^{}]*?"score"\s*:\s*(?P<score>null|-?\d+(?:\.\d+)?)',
    re.I | re.S,
)
_RATIONALE_RE = re.compile(r'"rationale"\s*:\s*"(?P<r>(?:[^"\\]|\\.)*)"', re.S)


def salvage_scores(text: str) -> dict[str, dict[str, Any]]:
    """Recover per-criterion scores from JSON a strict parser rejected.

    This exists because of a measurement bias, not for tidiness. Judges here are
    asked for JSON containing a free-text rationale, and a rationale that quotes
    the caller ("he said "hello" twice") produces invalid JSON. In the first
    funded run that hit the **transcript** judge on 4 of 6 calls and the audio
    judge on none of them -- so treating a parse failure as a missing score
    would have silently deleted most of one arm of a modality comparison and
    left the other intact. The scores are recoverable even when the rationales
    are not, and the scores are what the experiment is about.
    """
    out: dict[str, dict[str, Any]] = {}
    for m in _SCORE_RE.finditer(strip_fences(text)):
        raw = m.group("score")
        out[m.group("id").lower()] = {
            "score": None if raw.lower() == "null" else float(raw),
            "rationale": "",
            "salvaged": True,
        }
    return out
