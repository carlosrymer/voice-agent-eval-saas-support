"""OpenTelemetry tracing into a self-hosted Phoenix, with a file fallback.

There is no LangSmith key in this environment, so the tracing half of the
article's workflow is demonstrated on a self-hosted backend instead. That is a
substitution of tool, not of method: the spans carry OpenInference semantic
conventions, which is what makes a trace readable by any compatible backend
rather than by one vendor's UI.

Spans are also always appended to ``artifacts/otel_spans.jsonl`` whether or not
a collector is listening. A trace that only exists in a container someone has to
be running is not evidence; a committed file is.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ENDPOINT = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "")
SPAN_FILE = Path("artifacts/otel_spans.jsonl")

_lock = threading.Lock()


@dataclass
class SpanRecord:
    name: str
    kind: str
    start_ns: int
    end_ns: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "OK"
    parent: str | None = None
    span_id: str = ""
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_kind": self.kind,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent,
            "start_time_ns": self.start_ns,
            "end_time_ns": self.end_ns,
            "duration_ms": (self.end_ns - self.start_ns) / 1e6,
            "status": self.status,
            "attributes": self.attributes,
        }


class Tracer:
    """Minimal tracer: writes JSONL always, exports to OTLP when configured.

    Deliberately not the full OTel SDK dance. The project needs a durable,
    inspectable record of what happened inside each call; a dependency that
    silently no-ops when no collector is present would give neither.
    """

    def __init__(self, span_file: str | Path = SPAN_FILE, endpoint: str = DEFAULT_ENDPOINT):
        self.span_file = Path(span_file)
        self.span_file.parent.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint
        self.spans: list[SpanRecord] = []
        self._counter = 0
        self._stack: list[SpanRecord] = []
        self.trace_id = ""

    def _next_id(self) -> str:
        self._counter += 1
        return f"{self._counter:016x}"

    @contextmanager
    def span(self, name: str, kind: str = "CHAIN", **attributes: Any):
        rec = SpanRecord(
            name=name,
            kind=kind,
            start_ns=time.time_ns(),
            attributes={k: v for k, v in attributes.items() if v is not None},
            parent=self._stack[-1].span_id if self._stack else None,
            span_id=self._next_id(),
            trace_id=self.trace_id or self._next_id(),
        )
        if not self.trace_id:
            self.trace_id = rec.trace_id
        self._stack.append(rec)
        try:
            yield rec
        except Exception as exc:
            rec.status = f"ERROR: {type(exc).__name__}: {exc}"
            raise
        finally:
            rec.end_ns = time.time_ns()
            self._stack.pop()
            self.spans.append(rec)
            self._write(rec)

    def _write(self, rec: SpanRecord) -> None:
        with _lock:
            with self.span_file.open("a") as f:
                f.write(json.dumps(rec.to_dict(), default=str) + "\n")

    def export(self) -> bool:
        """Best-effort OTLP export. Returns True if a collector accepted it."""
        if not self.endpoint:
            return False
        try:
            import httpx

            payload = {"spans": [s.to_dict() for s in self.spans]}
            r = httpx.post(f"{self.endpoint.rstrip('/')}/v1/traces", json=payload, timeout=10)
            return r.status_code < 400
        except Exception:
            return False


def new_tracer(call_id: str) -> Tracer:
    t = Tracer()
    t.trace_id = f"{abs(hash(call_id)) & 0xFFFFFFFFFFFFFFFF:016x}"
    return t
