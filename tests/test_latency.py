"""Latency decomposition against calls whose timings I chose.

Every assertion here is against a number written into the fixture, not against
whatever the code produced the first time. The tolerance is one VAD analysis
frame plus a hop, because the caller-side boundary is measured off audio.
"""

from __future__ import annotations

import pytest

from tests.conftest import SyntheticCall
from voiceval.metrics.latency import aggregate, percentiles, turn_latencies

TOL_MS = 30.0


class TestPercentiles:
    def test_nearest_rank_returns_observed_values(self):
        p = percentiles([10.0, 20.0, 30.0, 40.0, 100.0])
        assert p.n == 5
        assert p.p50 == 30.0
        assert p.p95 == 100.0
        assert p.p99 == 100.0
        assert p.min == 10.0 and p.max == 100.0

    def test_single_observation(self):
        p = percentiles([42.0])
        assert (p.p50, p.p95, p.p99) == (42.0, 42.0, 42.0)

    def test_empty_sample_is_all_none_not_zero(self):
        """Zero would read as 'instant'; None reads as 'not measured'."""
        p = percentiles([])
        assert p.n == 0 and p.p50 is None and p.mean is None

    def test_nans_and_nones_are_dropped(self):
        p = percentiles([float("nan"), 5.0, None, 7.0])  # type: ignore[list-item]
        assert p.n == 2 and p.p50 == 5.0  # nearest-rank P50 of two values is the lower

    def test_p95_of_twenty_is_the_twentieth_value(self):
        p = percentiles([float(i) for i in range(1, 21)])
        assert p.p95 == 19.0  # ceil(0.95*20) = 19 -> index 18 -> value 19
        assert p.p99 == 20.0


class TestEndOfTurnLatency:
    @pytest.mark.parametrize("gap", [0.20, 0.62, 1.10, 2.00])
    def test_measured_gap_matches_the_scripted_gap(self, gap):
        c = SyntheticCall()
        c.caller_says(1.0, 1.5, "I need a credit")
        c.agent_says(2.5 + gap, 2.0, "sure", turn_started_t=2.5 + gap - 0.1)
        turns = turn_latencies(c.build())
        assert len(turns) == 1
        assert abs(turns[0].end_of_turn_ms - gap * 1000.0) < TOL_MS

    def test_agent_greeting_before_any_caller_speech_is_not_an_observation(self):
        """An unprompted opening line has no caller turn to be late relative to."""
        c = SyntheticCall()
        c.agent_says(0.2, 1.5, "Loopline support, how can I help?")
        c.caller_says(2.2, 1.0, "hi")
        c.agent_says(3.9, 1.5, "of course")
        turns = turn_latencies(c.build())
        assert len(turns) == 1, "only the reply to the caller should be measured"
        assert turns[0].turn_index == 1

    def test_latency_is_measured_from_speech_end_not_utterance_record(self):
        """The audio is the source of truth, even if bookkeeping disagrees."""
        c = SyntheticCall()
        c.caller_says(1.0, 2.0, "long question")
        c.agent_says(3.5, 1.0, "answer", turn_started_t=3.4)
        rec = c.build()
        rec.caller_utterances[0].end_t = 99.0  # deliberately corrupt the record
        turns = turn_latencies(rec)
        assert abs(turns[0].end_of_turn_ms - 500.0) < TOL_MS


class TestDecomposition:
    def test_components_always_sum_to_the_total(self):
        c = SyntheticCall()
        c.caller_says(0.5, 1.5, "please issue the credit")
        c.tool("send_verification_code", {"account_id": "acct_1042"}, requested_t=2.35,
               duration=0.40)
        c.tool("verify_identity", {"account_id": "acct_1042", "code": "418206"},
               requested_t=2.85, duration=0.30)
        c.agent_says(3.50, 2.0, "verified, issuing now", turn_started_t=2.20)
        tl = turn_latencies(c.build())[0]
        assert tl.unattributed_ms is not None
        assert abs(tl.attributed_ms() + tl.unattributed_ms - tl.end_of_turn_ms) < 1e-6

    def test_tool_time_is_exact_because_the_harness_ran_the_tool(self):
        c = SyntheticCall()
        c.caller_says(0.5, 1.0, "check my plan")
        c.tool("get_account", {"account_id": "acct_1042"}, requested_t=2.0, duration=0.250)
        c.tool("get_plan", {"plan_id": "pro"}, requested_t=2.30, duration=0.125)
        c.agent_says(2.8, 1.5, "you are on pro", turn_started_t=1.8)
        tl = turn_latencies(c.build())[0]
        assert tl.n_tool_calls == 2
        assert tl.tool_ms == pytest.approx(375.0, abs=1e-6)

    def test_asr_stage_uses_the_final_transcript_frame(self):
        c = SyntheticCall()
        c.caller_says(1.0, 1.0, "hello there", asr_delay=0.150)
        c.agent_says(2.7, 1.0, "hi", turn_started_t=2.5)
        tl = turn_latencies(c.build())[0]
        assert abs(tl.asr_ms - 150.0) < TOL_MS

    def test_asr_stage_is_none_when_provider_reports_no_transcripts(self):
        """Missing ASR time must not be relabelled as inference time.

        The stage that survives is `to_turn_start_ms`, which claims only what
        its two markers support: caller stopped, then the provider started
        responding. It does not assert how that interval was spent.
        """
        caps = dict(SyntheticCall().caps)
        caps["emits_caller_transcript"] = False
        c = SyntheticCall(caps=caps)
        c.caller_says(1.0, 1.0, "hello", asr_delay=None)
        c.agent_says(2.6, 1.0, "hi", turn_started_t=2.4)
        tl = turn_latencies(c.build())[0]
        assert tl.asr_ms is None
        assert "cannot be separated from inference" in tl.missing_reasons["asr_ms"]
        assert tl.to_turn_start_ms is not None
        assert abs(tl.attributed_ms() + tl.unattributed_ms - tl.end_of_turn_ms) < 1e-6

    def test_no_turn_start_frame_leaves_the_whole_gap_unattributed(self):
        """A provider that reports nothing gets a total and an honest shrug."""
        caps = dict(SyntheticCall().caps)
        caps["emits_caller_transcript"] = False
        caps["emits_turn_start"] = False
        c = SyntheticCall(caps=caps)
        c.caller_says(1.0, 1.0, "hello", asr_delay=None)
        c.agent_says(2.6, 1.0, "hi", turn_started_t=None, emit_events=True)
        tl = turn_latencies(c.build())[0]
        assert tl.asr_ms is None and tl.to_turn_start_ms is None
        assert "no turn-start frame" in tl.missing_reasons["to_turn_start_ms"]
        # to_first_audio_ms spans caller-end to first audio, i.e. everything.
        assert abs(tl.to_first_audio_ms - tl.end_of_turn_ms) < 1e-6
        assert tl.unattributed_ms == pytest.approx(0.0, abs=1e-6)

    def test_no_tool_turn_uses_the_single_synthesis_stage(self):
        c = SyntheticCall()
        c.caller_says(1.0, 1.0, "what is my limit")
        c.agent_says(2.7, 1.0, "fifty thousand", turn_started_t=2.4)
        tl = turn_latencies(c.build())[0]
        assert tl.tool_ms is None and tl.inference_pre_tool_ms is None
        assert tl.to_first_audio_ms == pytest.approx(300.0, abs=1.0)

    def test_model_time_between_tool_calls_is_not_charged_to_the_tools(self):
        """Two 100 ms tools 500 ms apart is 200 ms of tool time, not 500."""
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "do both")
        c.tool("send_verification_code", {"account_id": "a"}, requested_t=1.5, duration=0.1)
        c.tool("verify_identity", {"account_id": "a", "code": "1"}, requested_t=2.0, duration=0.1)
        c.agent_says(2.4, 1.0, "done", turn_started_t=1.3)
        tl = turn_latencies(c.build())[0]
        assert tl.tool_ms == pytest.approx(200.0)
        assert tl.inter_tool_inference_ms == pytest.approx(400.0, abs=1.0)
        assert abs(tl.attributed_ms() + tl.unattributed_ms - tl.end_of_turn_ms) < 1e-6

    def test_tool_turn_splits_inference_around_the_tool(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "look it up")
        c.tool("get_account", {"account_id": "a"}, requested_t=1.60, duration=0.20)
        c.agent_says(2.20, 1.0, "found it", turn_started_t=1.40)
        tl = turn_latencies(c.build())[0]
        assert tl.inference_pre_tool_ms is not None
        assert tl.tool_ms == pytest.approx(200.0)
        assert tl.inference_post_tool_ms == pytest.approx(400.0, abs=1.0)

    def test_customer_side_tools_do_not_count_against_agent_latency(self):
        """`check_inbox` runs in the caller's own workspace, not the agent's."""
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "reading the code now")
        c.tool("check_inbox", {}, requested_t=1.2, duration=5.0, requestor="user")
        c.agent_says(1.8, 1.0, "thanks", turn_started_t=1.5)
        tl = turn_latencies(c.build())[0]
        assert tl.n_tool_calls == 0 and tl.tool_ms is None


class TestAggregate:
    def test_unattributed_share_is_reported(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "a")
        c.agent_says(1.8, 1.0, "b", turn_started_t=1.7)
        rep = aggregate(turn_latencies(c.build()))
        assert 0.0 <= rep.unattributed_share <= 1.0

    def test_stage_means_sum_to_mean_end_of_turn_latency(self):
        """The stacked breakdown chart must add up to the headline bar."""
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "a")
        c.tool("get_account", {"account_id": "x"}, requested_t=1.4, duration=0.2)
        c.agent_says(1.9, 1.0, "b", turn_started_t=1.2)
        c.caller_says(3.2, 1.0, "c")
        c.agent_says(4.9, 1.0, "d", turn_started_t=4.6)
        rep = aggregate(turn_latencies(c.build()))
        assert sum(rep.stage_mean_ms().values()) == pytest.approx(rep.end_of_turn_ms.mean, abs=1e-6)

    def test_turn_counts(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "a")
        c.tool("get_account", {"account_id": "x"}, requested_t=1.3, duration=0.1)
        c.agent_says(1.6, 1.0, "b", turn_started_t=1.2)
        c.caller_says(3.0, 1.0, "c")
        c.agent_says(4.6, 1.0, "d", turn_started_t=4.4)
        rep = aggregate(turn_latencies(c.build()))
        assert rep.n_turns == 2 and rep.n_turns_with_tools == 1

    def test_empty_call_does_not_crash(self):
        rep = aggregate([])
        assert rep.n_turns == 0 and rep.end_of_turn_ms.p50 is None
        assert rep.unattributed_share is None


class TestBudgetInvariant:
    """The named stages plus the residual must equal the measured total.

    This is the claim the whole latency breakdown rests on, and it broke in a
    real run: a provider turn-start frame that landed outside the caller-stop to
    first-audio window was folded in anyway, producing a residual of -789.9 ms
    across the run — a negative slice of a stacked bar.
    """

    def test_a_turn_start_marker_after_first_audio_is_not_attributed(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hello")
        # turn_started_t deliberately *after* the audio begins.
        c.agent_says(2.0, 1.0, "hi", turn_started_t=2.5)
        tl = turn_latencies(c.build())[0]
        assert tl.to_turn_start_ms is None
        assert "outside" in tl.missing_reasons["to_turn_start_ms"]
        assert tl.unattributed_ms >= 0.0
        assert abs(tl.attributed_ms() + tl.unattributed_ms - tl.end_of_turn_ms) < 1e-6

    def test_a_tool_finishing_after_first_audio_cannot_overshoot_the_budget(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "look it up")
        # Tool result lands well after the agent has already started speaking.
        c.tool("get_account", {"account_id": "a"}, requested_t=1.4, duration=5.0)
        c.agent_says(1.9, 1.0, "found it", turn_started_t=1.6)
        tl = turn_latencies(c.build())[0]
        assert tl.unattributed_ms >= 0.0
        assert tl.attributed_ms() <= tl.end_of_turn_ms + 1e-6

    def test_residual_is_never_negative_across_a_messy_call(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.2, "a")
        c.tool("t1", {}, requested_t=1.5, duration=0.3)
        c.tool("t2", {}, requested_t=2.9, duration=4.0)
        c.agent_says(2.2, 1.5, "b", turn_started_t=3.9)
        for tl in turn_latencies(c.build()):
            assert tl.unattributed_ms is None or tl.unattributed_ms >= 0.0

    def test_stage_means_still_sum_when_some_turns_lack_a_total(self):
        """A turn with markers but no audio must not inflate the breakdown."""
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "a")
        c.agent_says(1.8, 1.0, "b", turn_started_t=1.7)
        c.caller_says(4.0, 1.0, "c")
        rec = c.build()
        # A turn the agent never actually voiced.
        from voiceval.metrics.timeline import AgentUtterance
        rec.agent_utterances.append(
            AgentUtterance(index=9, text="", audio_start_t=None, audio_end_t=None,
                           turn_started_t=5.5, completed_t=5.6)
        )
        rep = aggregate(turn_latencies(rec))
        assert sum(rep.stage_mean_ms().values()) == pytest.approx(
            rep.end_of_turn_ms.mean, abs=1e-6
        )
