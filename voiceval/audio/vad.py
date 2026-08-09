"""Energy-based voice activity detection with hysteresis.

Why roll this rather than pull in WebRTC VAD or Silero: every Experience number
in this project is derived from segment boundaries, so the detector has to be
something I can state the behaviour of exactly and test to the millisecond
against synthesised input. A neural VAD would be more robust on real noisy
telephony and less legible here -- and the audio in this harness is clean
datacenter PCM from a WebSocket, which is exactly the easy case. That tradeoff
is a limitation of the study, and it is written down in the README rather than
hidden in this docstring.

The detector is deliberately boring:

* frame the signal (20 ms window, 10 ms hop) and take RMS in dBFS
* pick a threshold from the signal's own noise floor, clamped both ways
* run a two-sided hysteresis state machine so a single dropout inside a word
  does not split one utterance into two

Segment boundaries are reported at the *edges of the triggering frames*, not at
the point the state machine became confident. Confirmation needs N consecutive
frames, so reporting the confirmation point would bias every onset late by a
fixed N*hop, which then shows up as a systematic error in end-of-turn latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from voiceval.audio.pcm import PCM

DEFAULT_FRAME_MS = 20.0
DEFAULT_HOP_MS = 10.0


@dataclass(frozen=True, order=True)
class Segment:
    """A half-open speech interval [start_s, end_s) on one track."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def overlap_s(self, other: "Segment") -> float:
        return max(0.0, min(self.end_s, other.end_s) - max(self.start_s, other.start_s))

    def intersects(self, other: "Segment") -> bool:
        return self.overlap_s(other) > 0.0


@dataclass(frozen=True)
class VadConfig:
    frame_ms: float = DEFAULT_FRAME_MS
    hop_ms: float = DEFAULT_HOP_MS
    #: Never treat anything quieter than this as speech, whatever the noise floor.
    abs_floor_dbfs: float = -45.0
    #: Speech must sit this far above the estimated noise floor.
    margin_db: float = 12.0
    #: Percentile of frame energies taken as the noise floor.
    noise_percentile: float = 10.0
    #: Safety clamp: the threshold is never allowed above (p95 energy - this),
    #: so a recording that is speech end-to-end still yields one segment
    #: instead of silence.
    loud_headroom_db: float = 6.0
    #: Consecutive above-threshold frames needed to open a segment.
    onset_frames: int = 2
    #: Consecutive below-threshold frames needed to close one (hangover).
    offset_frames: int = 5
    #: Segments shorter than this are discarded as clicks.
    min_speech_ms: float = 60.0
    #: Gaps shorter than this are absorbed into the surrounding segment.
    min_gap_ms: float = 120.0


@dataclass(frozen=True)
class VadResult:
    segments: list[Segment]
    threshold_dbfs: float
    noise_floor_dbfs: float
    frame_dbfs: np.ndarray = field(repr=False)
    frame_times_s: np.ndarray = field(repr=False)
    config: VadConfig = VadConfig()

    @property
    def speech_s(self) -> float:
        return sum(s.duration_s for s in self.segments)

    def speaking_at(self, t_s: float) -> bool:
        return any(s.start_s <= t_s < s.end_s for s in self.segments)


def frame_energies(pcm: PCM, cfg: VadConfig = VadConfig()) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame_dbfs, frame_start_times_s)."""
    x = pcm.samples()
    frame_n = max(1, int(round(cfg.frame_ms * pcm.sample_rate_hz / 1000.0)))
    hop_n = max(1, int(round(cfg.hop_ms * pcm.sample_rate_hz / 1000.0)))
    if x.size < frame_n:
        if x.size == 0:
            return np.zeros(0), np.zeros(0)
        x = np.pad(x, (0, frame_n - x.size))
    n_frames = 1 + (x.size - frame_n) // hop_n
    idx = np.arange(frame_n)[None, :] + hop_n * np.arange(n_frames)[:, None]
    frames = x[idx]
    power = np.mean(np.square(frames.astype(np.float64)), axis=1)
    with np.errstate(divide="ignore"):
        dbfs = 10.0 * np.log10(power)
    times = hop_n * np.arange(n_frames) / pcm.sample_rate_hz
    return dbfs, times


def choose_threshold(dbfs: np.ndarray, cfg: VadConfig) -> tuple[float, float]:
    """Return (threshold_dbfs, noise_floor_dbfs)."""
    finite = dbfs[np.isfinite(dbfs)]
    if finite.size == 0:
        return cfg.abs_floor_dbfs, float("-inf")
    noise = float(np.percentile(finite, cfg.noise_percentile))
    thresh = max(cfg.abs_floor_dbfs, noise + cfg.margin_db)
    loud = float(np.percentile(finite, 95.0))
    ceiling = loud - cfg.loud_headroom_db
    if thresh > ceiling:
        # Whole-file speech, or a noise floor estimate contaminated by speech.
        thresh = max(cfg.abs_floor_dbfs, ceiling)
    return thresh, noise


def detect(pcm: PCM, cfg: VadConfig = VadConfig()) -> VadResult:
    dbfs, times = frame_energies(pcm, cfg)
    if dbfs.size == 0:
        return VadResult([], cfg.abs_floor_dbfs, float("-inf"), dbfs, times, cfg)

    thresh, noise = choose_threshold(dbfs, cfg)
    hot = dbfs > thresh
    frame_dur_s = cfg.frame_ms / 1000.0

    segments: list[Segment] = []
    in_speech = False
    run_hot = 0
    run_cold = 0
    seg_start_i = 0
    last_hot_i = 0

    for i, is_hot in enumerate(hot):
        if is_hot:
            run_cold = 0
            run_hot += 1
            last_hot_i = i
            if not in_speech and run_hot >= cfg.onset_frames:
                in_speech = True
                seg_start_i = i - (cfg.onset_frames - 1)
        else:
            run_hot = 0
            if in_speech:
                run_cold += 1
                if run_cold >= cfg.offset_frames:
                    segments.append(
                        Segment(float(times[seg_start_i]), float(times[last_hot_i]) + frame_dur_s)
                    )
                    in_speech = False
                    run_cold = 0
    if in_speech:
        segments.append(Segment(float(times[seg_start_i]), float(times[last_hot_i]) + frame_dur_s))

    segments = _merge_gaps(segments, cfg.min_gap_ms / 1000.0)
    segments = [s for s in segments if s.duration_s >= cfg.min_speech_ms / 1000.0]
    return VadResult(segments, thresh, noise, dbfs, times, cfg)


def _merge_gaps(segments: list[Segment], min_gap_s: float) -> list[Segment]:
    if not segments:
        return []
    out = [segments[0]]
    for s in segments[1:]:
        prev = out[-1]
        if s.start_s - prev.end_s < min_gap_s:
            out[-1] = Segment(prev.start_s, max(prev.end_s, s.end_s))
        else:
            out.append(s)
    return out


def overlap_segments(a: list[Segment], b: list[Segment]) -> list[Segment]:
    """Intervals where a track-A segment and a track-B segment are both active."""
    out: list[Segment] = []
    for sa in a:
        for sb in b:
            lo, hi = max(sa.start_s, sb.start_s), min(sa.end_s, sb.end_s)
            if hi > lo:
                out.append(Segment(lo, hi))
    return _merge_gaps(sorted(out), 0.0)


def gaps_between(segments: list[Segment], until_s: float | None = None) -> list[Segment]:
    """Silent intervals strictly between consecutive segments.

    Leading and trailing silence are excluded on purpose: a caller who takes two
    seconds to start talking is not conversational friction, and trailing
    silence is an artefact of when the recording was stopped.
    """
    out: list[Segment] = []
    for prev, nxt in zip(segments, segments[1:]):
        if nxt.start_s > prev.end_s:
            out.append(Segment(prev.end_s, nxt.start_s))
    if until_s is not None and segments and until_s > segments[-1].end_s:
        pass  # trailing silence deliberately not counted; see docstring
    return out
