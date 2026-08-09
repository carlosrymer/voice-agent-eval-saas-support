"""Mono 16-bit little-endian PCM helpers.

Every audio buffer that crosses a module boundary in this project is a
:class:`PCM`: raw signed 16-bit little-endian samples plus the sample rate they
were captured at. Providers disagree about rates -- Gemini Live takes 16 kHz in
and returns 24 kHz out -- so the rate travels with the bytes rather than living
in a constant somewhere. Nothing here depends on a codec library; a call
recording is a `.wav` written by the stdlib `wave` module.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np

SAMPLE_WIDTH_BYTES = 2
INT16_FULL_SCALE = 32767.0


@dataclass(frozen=True)
class PCM:
    """Immutable mono signed-16-bit audio."""

    data: bytes
    sample_rate_hz: int

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {self.sample_rate_hz}")
        if len(self.data) % SAMPLE_WIDTH_BYTES:
            raise ValueError(
                f"PCM byte length {len(self.data)} is not a whole number of "
                f"{SAMPLE_WIDTH_BYTES}-byte samples"
            )

    @property
    def n_samples(self) -> int:
        return len(self.data) // SAMPLE_WIDTH_BYTES

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate_hz

    def samples(self) -> np.ndarray:
        """Samples as float32 in [-1, 1]."""
        if not self.data:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(self.data, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE

    @classmethod
    def from_samples(cls, samples: np.ndarray, sample_rate_hz: int) -> "PCM":
        """Build from float samples in [-1, 1], clipping rather than wrapping.

        Clipping matters: `np.float32 -> int16` on an out-of-range value wraps
        around in numpy, which turns a loud passage into a burst of noise that
        a VAD then reads as speech. Every synthesis path in this project goes
        through here so that failure mode cannot happen quietly.
        """
        clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        return cls((clipped * INT16_FULL_SCALE).astype("<i2").tobytes(), sample_rate_hz)

    @classmethod
    def silence(cls, duration_s: float, sample_rate_hz: int) -> "PCM":
        n = max(0, int(round(duration_s * sample_rate_hz)))
        return cls(b"\x00\x00" * n, sample_rate_hz)

    def slice_s(self, start_s: float, end_s: float) -> "PCM":
        a = max(0, int(round(start_s * self.sample_rate_hz)))
        b = min(self.n_samples, int(round(end_s * self.sample_rate_hz)))
        if b < a:
            b = a
        return PCM(self.data[a * SAMPLE_WIDTH_BYTES : b * SAMPLE_WIDTH_BYTES], self.sample_rate_hz)

    def __add__(self, other: "PCM") -> "PCM":
        if other.sample_rate_hz != self.sample_rate_hz:
            raise ValueError(
                f"cannot concatenate {self.sample_rate_hz} Hz with {other.sample_rate_hz} Hz; "
                "resample first"
            )
        return PCM(self.data + other.data, self.sample_rate_hz)


def concat(chunks: list[PCM], sample_rate_hz: int) -> PCM:
    """Concatenate chunks, tolerating an empty list."""
    out = bytearray()
    for c in chunks:
        if c.sample_rate_hz != sample_rate_hz:
            c = resample(c, sample_rate_hz)
        out += c.data
    return PCM(bytes(out), sample_rate_hz)


def resample(pcm: PCM, target_rate_hz: int) -> PCM:
    """Linear-interpolation resample.

    Deliberately simple. This is used to put the caller track and the agent
    track on a common clock for overlap arithmetic, not to produce audio
    anybody listens to critically. Anti-aliasing would matter if the metrics
    read spectra; they read energy envelopes, which linear interpolation
    preserves well enough at the 16k<->24k ratios in play.
    """
    if pcm.sample_rate_hz == target_rate_hz:
        return pcm
    src = pcm.samples()
    if src.size == 0:
        return PCM(b"", target_rate_hz)
    n_out = max(1, int(round(src.size * target_rate_hz / pcm.sample_rate_hz)))
    # Sample positions in source index space, endpoint-inclusive.
    x_out = np.linspace(0.0, src.size - 1, n_out, dtype=np.float64)
    return PCM.from_samples(np.interp(x_out, np.arange(src.size), src), target_rate_hz)


def mix(a: PCM, b: PCM, sample_rate_hz: int | None = None) -> PCM:
    """Sum two tracks into one, zero-padding the shorter to match."""
    rate = sample_rate_hz or a.sample_rate_hz
    xa, xb = resample(a, rate).samples(), resample(b, rate).samples()
    n = max(xa.size, xb.size)
    out = np.zeros(n, dtype=np.float32)
    out[: xa.size] += xa
    out[: xb.size] += xb
    return PCM.from_samples(out, rate)


def place(base: PCM, insert: PCM, at_s: float) -> PCM:
    """Overlay `insert` onto `base` starting at `at_s`, extending if needed."""
    rate = base.sample_rate_hz
    x = base.samples()
    y = resample(insert, rate).samples()
    start = int(round(at_s * rate))
    n = max(x.size, start + y.size)
    out = np.zeros(n, dtype=np.float32)
    out[: x.size] = x
    out[start : start + y.size] += y
    return PCM.from_samples(out, rate)


def write_wav(path: str, pcm: PCM) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH_BYTES)
        w.setframerate(pcm.sample_rate_hz)
        w.writeframes(pcm.data)


def to_wav_bytes(pcm: PCM) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH_BYTES)
        w.setframerate(pcm.sample_rate_hz)
        w.writeframes(pcm.data)
    return buf.getvalue()


def read_wav(path: str) -> PCM:
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1:
            raise ValueError("only mono WAV is supported")
        if w.getsampwidth() != SAMPLE_WIDTH_BYTES:
            raise ValueError("only 16-bit WAV is supported")
        return PCM(w.readframes(w.getnframes()), w.getframerate())


def rms_dbfs(samples: np.ndarray) -> float:
    """RMS level in dBFS. Silence returns -inf rather than raising."""
    if samples.size == 0:
        return float("-inf")
    r = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if r <= 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(r))
