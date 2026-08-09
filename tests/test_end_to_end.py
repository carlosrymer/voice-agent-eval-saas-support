"""The whole pipeline, offline, with no API key and no spend.

This drives the *real* orchestrator against the real tau2 environment, executing
real domain tools, assembling a real playout buffer, and then running the real
Execution and Outcome scorers over the result. Only the model is simulated.

It is the test that catches the class of bug the unit tests cannot: the pieces
being individually correct and wrongly wired. Both of the worst defects found
while building this -- the caller simulator playing the agent's side of the
conversation, and the agent's speech being reconstructed from a stale global
transcript so every turn got identical text -- were wiring bugs of exactly this
shape, invisible to a unit test and obvious end to end.
"""

from __future__ import annotations

import pytest

from voiceval.caller.simulator import BargeInPlan, CallerAction
from voiceval.domain import execute_user_tool, task_by_id
from voiceval.metrics.bargein import detect_barge_ins
from voiceval.metrics.friction import friction_report
from voiceval.metrics.latency import turn_latencies
from voiceval.orchestrator import CallConfig, Playout, run_call
from voiceval.providers.mock import MockScript, MockToolCall, MockTurn, MockVoiceProvider
from voiceval.scoring.execution import score_execution
from voiceval.scoring.outcome import score_outcome
from voiceval.tts import FixtureTTS

ACCOUNT = "acct_1042"


class ScriptedCaller:
    """A caller with no brain: it says what it is told, in order."""

    def __init__(self, lines: list[str], env=None, inbox_at: int | None = None):
        self.lines = list(lines)
        self.env = env
        self.inbox_at = inbox_at
        self.i = 0
        self.heard: list[str] = []
        self.tool_calls_made: list[tuple] = []
        self.ended = None

    def observe_agent(self, text: str) -> None:
        self.heard.append(text)

    def next_action(self) -> CallerAction:
        if self.inbox_at is not None and self.i == self.inbox_at and self.env is not None:
            r = execute_user_tool(self.env, "check_inbox", {})
            self.tool_calls_made.append(("check_inbox", {}, r.content))
            self.inbox_at = None
        if self.i >= len(self.lines):
            self.ended = "###STOP###"
            return CallerAction("end", terminator="###STOP###")
        text = self.lines[self.i]
        self.i += 1
        return CallerAction("speak", text=text)


def caller_factory(lines, inbox_at=None):
    def make(task, env, client, cfg):
        return ScriptedCaller(lines, env=env, inbox_at=inbox_at)
    return make


def verify_then_credit_script() -> MockScript:
    """A scripted agent that does T01 correctly: verify identity, then credit."""
    return MockScript(
        turns=[
            MockTurn(text="Loopline support, how can I help?", speech_duration_s=1.6),
            MockTurn(
                text="Let me send a verification code to the account owner.",
                tool_calls=[MockToolCall("send_verification_code", {"account_id": ACCOUNT})],
                speech_duration_s=2.0,
            ),
            MockTurn(
                text="Thank you, that code checks out.",
                tool_calls=[
                    MockToolCall("verify_identity", {"account_id": ACCOUNT, "code": "418206"})
                ],
                speech_duration_s=1.8,
            ),
            MockTurn(
                text="I have applied a three hundred dollar credit to your account.",
                tool_calls=[
                    MockToolCall(
                        "issue_account_credit",
                        {"account_id": ACCOUNT, "amount_cents": 30000,
                         "reason_code": "service_outage"},
                    )
                ],
                speech_duration_s=2.6,
            ),
        ]
    )


CALLER_LINES = [
    "Hi, this is Priya Raman from Northwind Labs.",
    "We lost six hours of campaign sending yesterday and I would like a three hundred dollar credit.",
    "The code is four one eight two zero six.",
    "Thank you, that is all.",
]


async def run_offline(script: MockScript, lines=None, inbox_at=2, **cfg_kw):
    task = task_by_id("T01_credit_within_cap")
    provider = MockVoiceProvider(script)
    cfg = CallConfig(
        max_turns=len(lines or CALLER_LINES),
        first_frame_s=30.0,
        turn_idle_s=5.0,
        max_playout_wait_s=0.0,  # virtual clock: no real waiting needed
        **cfg_kw,
    )
    return await run_call(
        task, provider, FixtureTTS(), client=None, config=cfg,
        caller_factory=caller_factory(lines or CALLER_LINES, inbox_at=inbox_at),
    )


class TestPlayout:
    """The playout buffer is the model of what the caller actually hears."""

    def test_contiguous_chunks_play_back_to_back_not_at_arrival_time(self):
        p = Playout(24000)
        from voiceval.audio.pcm import PCM

        chunk = PCM.silence(0.5, 24000)
        p.add(1.0, chunk)   # arrives at 1.0, plays 1.0-1.5
        p.add(1.05, chunk)  # arrives early, must queue behind the first
        assert p.chunks[1][0] == pytest.approx(1.5)
        assert p.last_end == pytest.approx(2.0)
        assert p.underruns == 0

    def test_a_stall_past_the_buffer_is_an_underrun_the_caller_hears(self):
        p = Playout(24000)
        from voiceval.audio.pcm import PCM

        p.add(1.0, PCM.silence(0.2, 24000))  # plays 1.0-1.2
        p.add(2.0, PCM.silence(0.2, 24000))  # nothing to play 1.2-2.0
        assert p.underruns == 1
        assert p.chunks[1][0] == pytest.approx(2.0)

    def test_interrupt_discards_the_unplayed_remainder(self):
        p = Playout(24000)
        from voiceval.audio.pcm import PCM

        for i in range(10):
            p.add(1.0 + i * 0.1, PCM.silence(0.1, 24000))  # 1.0 -> 2.0
        p.interrupt(1.45)
        assert p.last_end == pytest.approx(1.45, abs=1e-3)

    def test_interrupt_before_any_audio_leaves_nothing(self):
        p = Playout(24000)
        from voiceval.audio.pcm import PCM

        p.add(1.0, PCM.silence(0.5, 24000))
        p.interrupt(0.5)
        assert p.chunks == []


class TestOfflineCall:
    async def test_the_call_runs_and_records_both_tracks(self):
        rec = await run_offline(verify_then_credit_script())
        assert rec.caller_track is not None and rec.agent_track is not None
        assert rec.caller_track.n_samples > 0 and rec.agent_track.n_samples > 0
        assert len(rec.caller_utterances) == len(CALLER_LINES)
        assert rec.errors == []

    async def test_every_agent_turn_gets_its_own_text(self):
        """The regression guard for the stale-transcript bug."""
        rec = await run_offline(verify_then_credit_script())
        texts = [a.text for a in rec.agent_utterances if a.text]
        assert len(texts) >= 3
        assert len(set(texts)) == len(texts), f"duplicated agent turns: {texts}"

    async def test_tools_actually_executed_against_the_real_environment(self):
        rec = await run_offline(verify_then_credit_script())
        names = [t.name for t in rec.tool_executions if t.requestor == "assistant"]
        assert names == [
            "send_verification_code",
            "verify_identity",
            "issue_account_credit",
        ]
        assert all(t.ok for t in rec.tool_executions)
        assert any("verified" in (t.result or "").lower() for t in rec.tool_executions)

    async def test_latency_observations_are_produced_and_always_balance(self):
        rec = await run_offline(verify_then_credit_script())
        tls = turn_latencies(rec)
        assert tls, "a completed call must yield at least one latency observation"
        assert any(t.end_of_turn_ms and t.end_of_turn_ms > 0 for t in tls)
        for t in tls:
            if t.end_of_turn_ms is None:
                continue
            assert abs(t.attributed_ms() + t.unattributed_ms - t.end_of_turn_ms) < 1e-6

    async def test_scoring_runs_end_to_end_on_a_correct_call(self):
        rec = await run_offline(verify_then_credit_script())
        ex = score_execution(rec)
        oc = score_outcome(rec)
        assert ex.violations == [], f"unexpected violations: {ex.violations}"
        assert oc.error is None
        assert oc.reward == 1.0, f"expected the task to pass, got {oc.to_dict()}"
        assert ex.clean is True

    async def test_a_policy_breaking_agent_is_caught_end_to_end(self):
        """Same pipeline, an agent that credits without verifying identity."""
        script = MockScript(
            turns=[
                MockTurn(text="Loopline support.", speech_duration_s=1.2),
                MockTurn(
                    text="Sure, applying that credit now.",
                    tool_calls=[
                        MockToolCall(
                            "issue_account_credit",
                            {"account_id": ACCOUNT, "amount_cents": 30000,
                             "reason_code": "service_outage"},
                        )
                    ],
                    speech_duration_s=2.0,
                ),
            ]
        )
        rec = await run_offline(script, lines=CALLER_LINES[:2], inbox_at=None)
        ex = score_execution(rec)
        assert "P1" in {v.rule for v in ex.violations}
        assert ex.clean is False
        # ...and the outcome can still pass, which is the entire reason the two
        # axes are scored separately.
        assert score_outcome(rec).reward == 1.0

    async def test_friction_and_barge_in_run_on_a_real_recording(self):
        rec = await run_offline(verify_then_credit_script())
        fr = friction_report(rec)
        assert fr.n_caller_turns > 0 and fr.n_agent_turns > 0
        assert fr.caller_wer == 0.0, "the mock echoes ground truth, so WER must be zero"
        bi = detect_barge_ins(rec)
        assert bi.n_barge_ins == 0  # no barge-in scripted in this call

    async def test_a_scripted_barge_in_lands_and_is_measured(self):
        script = MockScript(
            turns=[
                MockTurn(text="Loopline support, how can I help?", speech_duration_s=1.5),
                MockTurn(text="Let me read you the whole outage policy in detail now.",
                         speech_duration_s=5.0, yield_latency_s=0.2),
                MockTurn(text="Of course, what would you like instead?", speech_duration_s=2.0),
                MockTurn(text="Understood, goodbye.", speech_duration_s=1.5),
            ]
        )
        rec = await run_offline(
            script, inbox_at=None, barge_in=BargeInPlan(turns=(1,), offset_s=1.0,
                                                        utterance="Sorry, hang on a second.")
        )
        assert any(u.is_barge_in for u in rec.caller_utterances)
        bi = detect_barge_ins(rec)
        assert bi.n_scripted_barge_ins == 1
        assert bi.n_barge_ins == 1, "the scripted interruption should have landed"
        assert bi.events[0].yielded is True

    async def test_the_record_round_trips_through_disk(self, tmp_path):
        rec = await run_offline(verify_then_credit_script())
        path = rec.save(tmp_path)
        from voiceval.metrics.timeline import CallRecord

        again = CallRecord.load(path)
        assert again.call_id == rec.call_id
        assert len(again.tool_executions) == len(rec.tool_executions)
        assert again.caller_track.n_samples == rec.caller_track.n_samples
        assert len(again.events) == len(rec.events), "the event stream must survive"
        # The metrics must give identical answers after a round trip, or the
        # committed artifacts do not reproduce the published numbers.
        assert [t.to_dict() for t in turn_latencies(again)] == [
            t.to_dict() for t in turn_latencies(rec)
        ]

    async def test_synthetic_runs_are_flagged_as_synthetic(self):
        """A mock run must never be presentable as a real one."""
        rec = await run_offline(verify_then_credit_script())
        assert rec.synthetic is True
        assert "synthetic" in rec.capabilities["notes"].lower()
