"""Synthetic audio with exactly known ground truth.

This module exists so the Experience detectors can be tested to the millisecond
without spending a cent on a model. Every generator here returns audio whose
speech boundaries I chose, and :class:`TrackBuilder` hands back both the audio
and the ground-truth segment list, so a test can assert "the detector found a
1.400 s gap" against a gap I actually placed at 1.400 s rather than against
whatever the detector happened to say the first time it ran.

The "speech" is not speech. It is a harmonic stack with a syllable-rate
amplitude envelope, which is enough to give an energy VAD the same onsets,
offsets and intra-word energy continuity that real speech does. Where a
detector's behaviour depends on spectral content rather than energy, that
detector cannot be honestly validated here and is not claimed to be.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from voiceval.audio.pcm import PCM, place
from voiceval.audio.vad import Segment

DEFAULT_RATE = 16000

#: Deterministic seed so a failing test is reproducible from the test name alone.
DEFAULT_SEED = 20260809


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(DEFAULT_SEED if seed is None else seed)


def silence(duration_s: float, rate: int = DEFAULT_RATE) -> PCM:
    return PCM.silence(duration_s, rate)


def noise(duration_s: float, level_dbfs: float = -60.0, rate: int = DEFAULT_RATE,
          seed: int | None = None) -> PCM:
    n = int(round(duration_s * rate))
    amp = 10.0 ** (level_dbfs / 20.0)
    return PCM.from_samples(_rng(seed).normal(0.0, amp, n), rate)


def tone(duration_s: float, freq_hz: float = 440.0, level_dbfs: float = -20.0,
         rate: int = DEFAULT_RATE) -> PCM:
    n = int(round(duration_s * rate))
    t = np.arange(n) / rate
    amp = 10.0 ** (level_dbfs / 20.0)
    return PCM.from_samples(amp * np.sin(2 * np.pi * freq_hz * t), rate)


def speech_like(
    duration_s: float,
    f0_hz: float = 120.0,
    level_dbfs: float = -20.0,
    syllable_hz: float = 4.0,
    envelope_floor: float = 0.45,
    rate: int = DEFAULT_RATE,
    seed: int | None = None,
) -> PCM:
    """A voiced-speech stand-in: harmonic stack under a syllable envelope.

    `envelope_floor` keeps the quietest point of the envelope well above the VAD
    threshold. Real speech does dip to silence between words; the point of this
    generator is to produce *one* utterance with unambiguous edges, and gaps
    between utterances are placed explicitly by the caller instead. A generator
    that randomly split itself in two would make every boundary assertion in the
    test suite conditional on the RNG.
    """
    n = max(1, int(round(duration_s * rate)))
    t = np.arange(n) / rate
    rng = _rng(seed)

    wave = np.zeros(n, dtype=np.float64)
    for k, weight in enumerate([1.0, 0.6, 0.4, 0.25, 0.15, 0.1], start=1):
        wave += weight * np.sin(2 * np.pi * f0_hz * k * t + rng.uniform(0, 2 * np.pi))
    # A little breath noise so the spectrum is not a pure line spectrum.
    wave += 0.08 * rng.normal(0.0, 1.0, n)

    env = envelope_floor + (1.0 - envelope_floor) * (
        0.5 * (1.0 - np.cos(2 * np.pi * syllable_hz * t))
    )
    # 5 ms raised-cosine edges: an instantaneous onset is a click, and a click
    # has energy in every frame it touches, which would smear the onset frame.
    edge_n = min(n // 2, int(round(0.005 * rate)))
    if edge_n > 1:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(edge_n) / edge_n))
        env[:edge_n] *= ramp
        env[-edge_n:] *= ramp[::-1]

    y = wave * env
    peak = float(np.max(np.abs(y))) or 1.0
    y = y / peak
    rms = float(np.sqrt(np.mean(np.square(y)))) or 1.0
    y = y * (10.0 ** (level_dbfs / 20.0)) / rms
    return PCM.from_samples(np.clip(y, -0.99, 0.99), rate)


@dataclass
class TrackBuilder:
    """Compose a track from utterances placed at exact offsets.

    Returns both the rendered audio and the ground truth, so tests assert the
    detector against the specification rather than against itself.
    """

    rate: int = DEFAULT_RATE
    noise_floor_dbfs: float | None = -62.0
    seed: int | None = None
    _placed: list[tuple[float, PCM, str]] = field(default_factory=list)

    def add(self, at_s: float, pcm: PCM, label: str = "utt") -> "TrackBuilder":
        self._placed.append((at_s, pcm, label))
        return self

    def say(self, at_s: float, duration_s: float, label: str = "utt", **kw) -> "TrackBuilder":
        kw.setdefault("rate", self.rate)
        kw.setdefault("seed", self.seed)
        return self.add(at_s, speech_like(duration_s, **kw), label)

    @property
    def truth(self) -> list[Segment]:
        return sorted(Segment(at, at + p.duration_s) for at, p, _ in self._placed)

    @property
    def labels(self) -> list[tuple[str, Segment]]:
        return [(lbl, Segment(at, at + p.duration_s)) for at, p, lbl in self._placed]

    def render(self, total_s: float | None = None) -> PCM:
        end = total_s if total_s is not None else max(
            (at + p.duration_s for at, p, _ in self._placed), default=0.0
        )
        base = (
            noise(end, self.noise_floor_dbfs, self.rate, self.seed)
            if self.noise_floor_dbfs is not None
            else silence(end, self.rate)
        )
        for at, pcm, _ in self._placed:
            base = place(base, pcm, at)
        return base
