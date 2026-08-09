"""VAD boundary accuracy against fixtures whose truth I placed by hand.

The contract these tests pin down is the one every Experience number depends
on: a speech burst placed at a known offset must come back with its onset and
offset within one analysis frame. If that drifts, silence-gap durations,
end-of-turn latency and barge-in yield latency all drift with it, silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from voiceval.audio import fixtures as fx
from voiceval.audio.pcm import PCM, mix, place, resample
from voiceval.audio.vad import Segment, VadConfig, detect, gaps_between, overlap_segments

#: One 20 ms frame plus one 10 ms hop. Boundaries are reported at frame edges,
#: so this is the tightest tolerance the analysis grid can support.
TOL_S = 0.030


def approx(a: float, b: float, tol: float = TOL_S) -> bool:
    return abs(a - b) <= tol


class TestFrameEnergies:
    def test_silence_is_detected_as_no_speech(self):
        r = detect(fx.silence(3.0))
        assert r.segments == []

    def test_empty_audio_is_handled(self):
        r = detect(PCM(b"", 16000))
        assert r.segments == []

    def test_low_level_noise_alone_is_not_speech(self):
        r = detect(fx.noise(3.0, level_dbfs=-62.0))
        assert r.segments == []

    def test_speech_is_far_above_the_noise_floor(self):
        t = fx.TrackBuilder().say(0.5, 1.0).render(3.0)
        r = detect(t)
        assert r.noise_floor_dbfs < r.threshold_dbfs < -20.0


class TestSingleUtterance:
    @pytest.mark.parametrize("start,dur", [(0.0, 1.0), (0.5, 2.0), (1.234, 0.75), (2.0, 0.2)])
    def test_boundaries_recovered_within_one_frame(self, start, dur):
        track = fx.TrackBuilder().say(start, dur).render(start + dur + 1.0)
        segs = detect(track).segments
        assert len(segs) == 1, f"expected one segment, got {segs}"
        assert approx(segs[0].start_s, start), f"onset {segs[0].start_s} vs {start}"
        assert approx(segs[0].end_s, start + dur), f"offset {segs[0].end_s} vs {start + dur}"

    def test_whole_file_speech_still_yields_one_segment(self):
        """The adaptive threshold must not read an all-speech file as silence."""
        segs = detect(fx.speech_like(2.0)).segments
        assert len(segs) == 1
        assert segs[0].duration_s > 1.9

    def test_quiet_utterance_below_absolute_floor_is_ignored(self):
        track = fx.TrackBuilder(noise_floor_dbfs=None).say(1.0, 1.0, level_dbfs=-55.0).render(3.0)
        assert detect(track).segments == []

    def test_click_shorter_than_min_speech_is_dropped(self):
        track = fx.TrackBuilder().add(1.0, fx.tone(0.02, level_dbfs=-10.0)).render(3.0)
        assert detect(track).segments == []


class TestGaps:
    @pytest.mark.parametrize("gap", [0.30, 0.80, 1.40, 2.50])
    def test_known_gap_duration_is_recovered(self, gap):
        b = fx.TrackBuilder().say(0.5, 1.0, label="a").say(0.5 + 1.0 + gap, 1.0, label="b")
        segs = detect(b.render(0.5 + 2.0 + gap + 0.5)).segments
        assert len(segs) == 2, f"gap={gap}: {segs}"
        gaps = gaps_between(segs)
        assert len(gaps) == 1
        assert approx(gaps[0].duration_s, gap), f"measured {gaps[0].duration_s:.3f}s vs {gap}s"

    def test_gap_shorter_than_min_gap_is_absorbed(self):
        b = fx.TrackBuilder().say(0.5, 0.6).say(0.5 + 0.6 + 0.05, 0.6)
        segs = detect(b.render(3.0)).segments
        assert len(segs) == 1, "a 50 ms gap must not split one utterance in two"

    def test_leading_and_trailing_silence_are_not_gaps(self):
        b = fx.TrackBuilder().say(2.0, 1.0)
        segs = detect(b.render(8.0)).segments
        assert gaps_between(segs) == []

    def test_three_utterances_give_two_gaps(self):
        b = fx.TrackBuilder().say(0.0, 0.5).say(1.5, 0.5).say(3.5, 0.5)
        gaps = gaps_between(detect(b.render(5.0)).segments)
        assert len(gaps) == 2
        assert approx(gaps[0].duration_s, 1.0)
        assert approx(gaps[1].duration_s, 1.5)


class TestOverlap:
    @pytest.mark.parametrize("overlap", [0.10, 0.30, 0.75])
    def test_cross_track_overlap_duration_is_exact(self, overlap):
        agent = fx.TrackBuilder().say(0.0, 2.0).render(5.0)
        caller = fx.TrackBuilder().say(2.0 - overlap, 1.5).render(5.0)
        segs = overlap_segments(detect(caller).segments, detect(agent).segments)
        assert len(segs) == 1
        assert approx(segs[0].duration_s, overlap), f"{segs[0].duration_s:.3f} vs {overlap}"

    def test_no_overlap_when_turns_are_clean(self):
        agent = fx.TrackBuilder().say(0.0, 2.0).render(6.0)
        caller = fx.TrackBuilder().say(2.5, 1.5).render(6.0)
        assert overlap_segments(detect(caller).segments, detect(agent).segments) == []

    def test_touching_turns_do_not_count_as_overlap(self):
        """Back-to-back turns share an instant, not an interval."""
        a, b = Segment(0.0, 1.0), Segment(1.0, 2.0)
        assert a.overlap_s(b) == 0.0
        assert overlap_segments([a], [b]) == []


class TestSampleRates:
    def test_boundaries_hold_at_24k_agent_rate(self):
        """Gemini Live returns 24 kHz; the caller mic is 16 kHz."""
        track = fx.TrackBuilder(rate=24000).say(1.0, 1.2).render(3.0)
        segs = detect(track).segments
        assert len(segs) == 1
        assert approx(segs[0].start_s, 1.0) and approx(segs[0].end_s, 2.2)

    def test_resample_preserves_duration_and_boundaries(self):
        track = fx.TrackBuilder(rate=24000).say(0.8, 1.0).render(3.0)
        down = resample(track, 16000)
        assert abs(down.duration_s - track.duration_s) < 0.001
        segs = detect(down).segments
        assert len(segs) == 1 and approx(segs[0].start_s, 0.8)


class TestRobustness:
    @pytest.mark.parametrize("floor", [-70.0, -62.0, -50.0])
    def test_boundaries_survive_a_range_of_noise_floors(self, floor):
        track = fx.TrackBuilder(noise_floor_dbfs=floor).say(1.0, 1.0).render(3.0)
        segs = detect(track).segments
        assert len(segs) == 1
        assert approx(segs[0].start_s, 1.0) and approx(segs[0].end_s, 2.0)

    def test_detection_is_deterministic(self):
        track = fx.TrackBuilder(seed=7).say(0.5, 1.0).render(3.0)
        assert detect(track).segments == detect(track).segments

    def test_summed_tracks_do_not_clip_into_false_speech(self):
        """`mix` must clip rather than wrap; a wrapped sum reads as loud noise."""
        loud = fx.speech_like(1.0, level_dbfs=-1.0)
        summed = mix(loud, loud)
        assert np.max(np.abs(summed.samples())) <= 1.0
