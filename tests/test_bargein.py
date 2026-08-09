"""Barge-in detection: fire on real interruptions, stay quiet on near-misses.

A detector that never fires reports a perfect yield rate and looks like good
news, so every positive case here is paired with a near-miss that must NOT
trigger it. The near-misses are the ones that catch a sloppy implementation:
back-to-back turns with no overlap, a caller starting 20 ms after the agent
stopped, the agent interrupting the caller instead of the other way round.
"""

from __future__ import annotations

import pytest

from tests.conftest import SyntheticCall
from voiceval.metrics.bargein import detect_barge_ins

TOL_MS = 35.0


class TestDetection:
    def test_caller_speaking_over_the_agent_is_a_barge_in(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.5, 3.0, "let me read out the whole policy to you")
        c.caller_says(2.5, 1.5, "no wait", barge_in=True)
        r = detect_barge_ins(c.build())
        assert r.n_barge_ins == 1
        assert abs(r.events[0].caller_onset_t - 2.5) < 0.03

    def test_clean_turn_taking_is_not_a_barge_in(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.5, 2.0, "sure")
        c.caller_says(4.0, 1.0, "thanks")
        assert detect_barge_ins(c.build()).n_barge_ins == 0

    def test_caller_starting_just_after_the_agent_stops_is_not_a_barge_in(self):
        """The tightest near-miss: 80 ms of gap, no overlap."""
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.5, 2.0, "sure")  # ends 3.5
        c.caller_says(3.58, 1.0, "ok")
        assert detect_barge_ins(c.build()).n_barge_ins == 0

    def test_agent_talking_over_a_still_speaking_caller_is_not_caller_barge_in(self):
        """The agent cutting the caller off is a different fault; not counted here."""
        c = SyntheticCall()
        c.caller_says(0.0, 3.0, "so what happened was, um, yesterday around noon")
        c.agent_says(2.0, 2.0, "let me stop you there")
        assert detect_barge_ins(c.build()).n_barge_ins == 0

    def test_two_separate_barge_ins_are_both_counted(self):
        c = SyntheticCall()
        c.caller_says(0.0, 0.8, "hi")
        c.agent_says(1.2, 3.0, "first long answer")
        c.caller_says(2.0, 0.8, "wait", barge_in=True)
        c.agent_says(5.0, 3.0, "second long answer")
        c.caller_says(6.0, 0.8, "stop", barge_in=True)
        assert detect_barge_ins(c.build()).n_barge_ins == 2


class TestYieldLatency:
    @pytest.mark.parametrize("yield_s", [0.10, 0.18, 0.45])
    def test_yield_latency_matches_the_scripted_value(self, yield_s):
        onset = 2.5
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        # Agent audio is cut short exactly `yield_s` after the caller starts.
        c.agent_says(1.5, (onset + yield_s) - 1.5, "truncated", interrupted=True)
        c.caller_says(onset, 1.5, "no wait", barge_in=True)
        ev = detect_barge_ins(c.build()).events[0]
        assert abs(ev.yield_latency_ms - yield_s * 1000.0) < TOL_MS
        assert ev.yielded is True

    def test_an_agent_that_never_stops_does_not_count_as_yielding(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.5, 5.0, "droning on regardless")
        c.caller_says(2.0, 1.0, "stop please", barge_in=True)
        ev = detect_barge_ins(c.build()).events[0]
        assert ev.yielded is False
        assert ev.yield_latency_ms > 800.0

    def test_yield_window_is_a_parameter_not_a_hidden_constant(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.5, 1.1, "cut short", interrupted=True)  # stops at 2.6
        c.caller_says(2.0, 1.0, "stop", barge_in=True)         # yields in ~600 ms
        rec = c.build()
        assert detect_barge_ins(rec, yield_window_ms=800.0).events[0].yielded is True
        assert detect_barge_ins(rec, yield_window_ms=300.0).events[0].yielded is False

    def test_overlap_duration_is_the_simultaneous_speech_only(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.5, 1.2, "cut", interrupted=True)  # stops at 2.7
        c.caller_says(2.4, 2.0, "hang on", barge_in=True)
        ev = detect_barge_ins(c.build()).events[0]
        assert abs(ev.overlap_ms - 300.0) < TOL_MS


class TestProviderSignal:
    def test_interrupt_frame_is_recorded_alongside_the_audio_measurement(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.5, 1.0, "cut", interrupted=True)  # INTERRUPTED emitted at 2.5
        c.caller_says(2.3, 1.0, "wait", barge_in=True)
        ev = detect_barge_ins(c.build()).events[0]
        assert ev.provider_signalled is True
        assert abs(ev.signal_latency_ms) < 60.0

    def test_a_signal_that_lies_about_the_audio_is_visible(self):
        """Control plane says stopped; the audio kept playing for 700 ms."""
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.5, 1.5, "still going", interrupted=False)  # audio to 3.0
        c.event.__self__  # noqa: B018  (readability: event() used below)
        from voiceval.providers.base import EventKind

        c.event(EventKind.INTERRUPTED, 2.3, message="caller barge-in")
        c.caller_says(2.2, 1.2, "wait", barge_in=True)
        ev = detect_barge_ins(c.build()).events[0]
        assert ev.provider_signalled is True
        # Negative: the frame arrived well before the audio actually stopped.
        assert ev.signal_latency_ms < -500.0

    def test_no_signal_still_yields_a_measurement(self):
        caps = dict(SyntheticCall().caps)
        caps["emits_interrupt_event"] = False
        c = SyntheticCall(caps=caps)
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.5, 0.7, "cut", emit_events=False)
        c.caller_says(2.0, 1.0, "wait", barge_in=True)
        ev = detect_barge_ins(c.build()).events[0]
        assert ev.provider_signalled is False
        assert ev.signal_latency_ms is None
        assert ev.yield_latency_ms > 0


class TestStateLoss:
    def test_restarting_the_interrupted_sentence_is_state_loss(self):
        line = "I can issue a three hundred dollar credit to your account today"
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.5, 1.0, line, interrupted=True)
        c.caller_says(2.3, 0.8, "sorry go on", barge_in=True)
        c.agent_says(4.0, 2.0, line)  # says the identical thing again
        ev = detect_barge_ins(c.build()).events[0]
        assert ev.restarted is True
        assert ev.state_preserved is False

    def test_carrying_on_with_new_content_preserves_state(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.5, 1.0, "I can issue a three hundred dollar credit", interrupted=True)
        c.caller_says(2.3, 0.8, "make it four", barge_in=True)
        c.agent_says(4.0, 2.0, "understood, four hundred needs my manager to sign off")
        ev = detect_barge_ins(c.build()).events[0]
        assert ev.restarted is False
        assert ev.state_preserved is True

    def test_a_tool_call_dropped_by_the_interruption_is_state_loss(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.dropped_tool("issue_account_credit", {"account_id": "a", "amount_cents": 30000},
                       requested_t=1.4)
        c.agent_says(1.6, 1.0, "issuing that now", interrupted=True)
        c.caller_says(2.3, 0.8, "hold on", barge_in=True)
        c.agent_says(4.0, 1.5, "of course, what would you like to change")
        ev = detect_barge_ins(c.build()).events[0]
        assert ev.dropped_tool_calls == ["call_0"]
        assert ev.state_preserved is False

    def test_a_completed_tool_call_is_not_a_dropped_one(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.tool("get_account", {"account_id": "a"}, requested_t=1.4, duration=0.2)
        c.agent_says(1.8, 1.0, "you are on the pro plan", interrupted=True)
        c.caller_says(2.5, 0.8, "hold on", barge_in=True)
        c.agent_says(4.0, 1.5, "sure, take your time")
        ev = detect_barge_ins(c.build()).events[0]
        assert ev.dropped_tool_calls == []
        assert ev.state_preserved is True


class TestReport:
    def test_scripted_but_unlanded_barge_ins_are_surfaced(self):
        """If a scripted interruption missed the agent's speech, say so.

        Otherwise a run that tested barge-in zero times reports a 100% yield
        rate on a sample of zero and reads like a pass.
        """
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.5, 1.0, "short answer")   # over by 2.5
        c.caller_says(3.0, 1.0, "wait", barge_in=True)  # too late; no overlap
        r = detect_barge_ins(c.build())
        assert r.n_barge_ins == 0
        assert r.n_scripted_barge_ins == 1

    def test_rates_are_none_rather_than_one_when_nothing_was_tested(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.5, 1.0, "ok")
        d = detect_barge_ins(c.build()).to_dict()
        assert d["yield_rate"] is None and d["state_preserved_rate"] is None
