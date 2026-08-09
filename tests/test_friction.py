"""Conversational friction metrics against calls with known properties."""

from __future__ import annotations

import pytest

from tests.conftest import SyntheticCall
from voiceval.metrics.friction import (
    friction_report,
    unmatched_question_rate,
    word_error_rate,
)

TOL_S = 0.035


class TestDeadAir:
    @pytest.mark.parametrize("gap", [1.2, 2.5, 4.0])
    def test_a_known_dead_air_gap_is_found_and_measured(self, gap):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.0 + gap, 1.0, "sorry for the wait")
        r = friction_report(c.build())
        assert r.n_long_silences == 1
        assert abs(r.longest_silence_s - gap) < TOL_S

    def test_a_pause_while_the_agent_is_still_talking_is_not_dead_air(self):
        """Dead air needs *nobody* speaking, not just one track idle."""
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.2, 4.0, "a long uninterrupted explanation")
        r = friction_report(c.build())
        assert r.n_long_silences == 0

    def test_gap_below_threshold_is_not_counted(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.6, 1.0, "yes")  # 600 ms gap
        assert friction_report(c.build()).n_long_silences == 0

    def test_threshold_is_a_parameter(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.8, 1.0, "yes")  # 800 ms gap
        rec = c.build()
        assert friction_report(rec, silence_threshold_s=1.0).n_long_silences == 0
        assert friction_report(rec, silence_threshold_s=0.5).n_long_silences == 1

    def test_trailing_recording_silence_is_not_dead_air(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.3, 1.0, "bye")
        r = friction_report(c.build(total_s=30.0))
        assert r.n_long_silences == 0


class TestOverlap:
    def test_overlap_total_matches_the_scripted_overlap(self):
        c = SyntheticCall()
        c.agent_says(1.0, 2.0, "talking")     # 1.0 - 3.0
        c.caller_says(2.5, 1.5, "over you")   # 2.5 - 4.0 -> 500 ms overlap
        r = friction_report(c.build())
        assert abs(r.overlap_total_s - 0.5) < TOL_S
        assert r.n_overlaps == 1

    def test_clean_call_reports_zero_overlap(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.4, 1.0, "hello")
        r = friction_report(c.build())
        assert r.overlap_total_s == 0.0 and r.n_overlaps == 0


class TestPhraseMatching:
    def test_agent_asking_the_caller_to_repeat_is_counted(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "my account is A B C one two three")
        c.agent_says(1.4, 1.0, "Sorry, could you repeat that one more time?")
        r = friction_report(c.build())
        assert r.agent_repeat_requests >= 1
        assert any(m["role"] == "agent" and m["kind"] == "repeat" for m in r.matched_phrases)

    def test_caller_asking_the_agent_to_repeat_is_counted_separately(self):
        c = SyntheticCall()
        c.agent_says(0.0, 1.0, "your limit is fifty thousand per day")
        c.caller_says(1.4, 1.0, "sorry, what?")
        r = friction_report(c.build())
        assert r.caller_repeat_requests >= 1 and r.agent_repeat_requests == 0

    def test_confirmation_is_a_clarification_not_a_repeat_request(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "three hundred dollars")
        c.agent_says(1.4, 1.0, "Just to confirm, that is three hundred dollars?")
        r = friction_report(c.build())
        assert r.agent_clarifications == 1 and r.agent_repeat_requests == 0

    def test_an_ordinary_question_is_not_friction(self):
        """The near-miss that matters: normal service questions must stay quiet."""
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.4, 1.5, "What is the email address on the account?")
        r = friction_report(c.build())
        assert r.agent_repeat_requests == 0 and r.agent_clarifications == 0
        assert r.unmatched_agent_questions == 1

    def test_unmatched_question_rate_bounds_the_phrase_list_recall(self):
        c = SyntheticCall()
        c.agent_says(0.0, 1.0, "Could you repeat that again?")
        c.agent_says(2.0, 1.0, "What is your account ID?")
        r = friction_report(c.build())
        assert unmatched_question_rate(r) == pytest.approx(0.5)

    def test_rate_is_none_when_the_agent_asked_nothing(self):
        c = SyntheticCall()
        c.agent_says(0.0, 1.0, "Your credit has been applied.")
        assert unmatched_question_rate(friction_report(c.build())) is None


class TestRepetition:
    def test_the_agent_saying_the_same_thing_twice_is_flagged(self):
        line = "I have applied a three hundred dollar credit to your account"
        c = SyntheticCall()
        c.agent_says(0.0, 1.5, line)
        c.caller_says(2.0, 1.0, "sorry the line dropped")
        c.agent_says(3.5, 1.5, line)
        r = friction_report(c.build())
        assert r.n_agent_repetitions == 1

    def test_different_content_is_not_repetition(self):
        c = SyntheticCall()
        c.agent_says(0.0, 1.5, "I have applied a three hundred dollar credit")
        c.agent_says(2.5, 1.5, "Is there anything else I can help with today?")
        assert friction_report(c.build()).n_agent_repetitions == 0

    def test_a_repeated_short_stock_phrase_is_not_over_flagged(self):
        """Politeness boilerplate differing in content must not read as a loop."""
        c = SyntheticCall()
        c.agent_says(0.0, 1.0, "One moment while I check your account balance.")
        c.agent_says(2.0, 1.0, "One moment while I raise this with my manager.")
        assert friction_report(c.build()).n_agent_repetitions == 0


class TestWordErrorRate:
    def test_perfect_transcription_is_zero(self):
        assert word_error_rate("issue a three hundred dollar credit",
                               "issue a three hundred dollar credit") == 0.0

    def test_one_wrong_word_in_six(self):
        assert word_error_rate("issue a three hundred dollar credit",
                               "issue a free hundred dollar credit") == pytest.approx(1 / 6)

    def test_deletions_and_insertions_count(self):
        assert word_error_rate("a b c d", "a c d") == pytest.approx(0.25)
        assert word_error_rate("a b c d", "a b x c d") == pytest.approx(0.25)

    def test_empty_hypothesis_is_total_failure(self):
        assert word_error_rate("hello there", "") == 1.0

    def test_case_and_punctuation_are_normalised_away(self):
        assert word_error_rate("Hello, there!", "hello there") == 0.0

    def test_call_level_wer_averages_over_utterances(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "acct one zero four two", asr="acct one zero four two")
        c.caller_says(2.0, 1.0, "priya at northwind labs",
                      asr="prea at northwind labs")
        r = friction_report(c.build())
        assert r.caller_wer_n_utterances == 2
        assert r.caller_wer == pytest.approx((0.0 + 0.25) / 2)

    def test_wer_is_none_when_the_provider_returned_no_asr(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hello", asr_delay=None)
        c.caller_utterances[0].asr_text = None
        assert friction_report(c.build()).caller_wer is None


class TestCadence:
    def test_turn_counts_and_speech_ratio(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "hi")
        c.agent_says(1.5, 2.0, "hello")
        c.caller_says(4.0, 1.0, "thanks")
        c.agent_says(5.5, 2.0, "bye")
        r = friction_report(c.build())
        assert r.n_caller_turns == 2 and r.n_agent_turns == 2
        assert r.speech_ratio_agent_to_caller == pytest.approx(2.0, abs=0.1)

    def test_early_termination_is_flagged_from_the_ended_reason(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "this is ridiculous, goodbye")
        c.ended_reason = "caller_hung_up"
        assert friction_report(c.build()).early_termination is True

    def test_normal_completion_is_not_early_termination(self):
        c = SyntheticCall()
        c.caller_says(0.0, 1.0, "thanks, bye")
        assert friction_report(c.build()).early_termination is False
