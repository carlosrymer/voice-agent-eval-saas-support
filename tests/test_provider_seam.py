"""The provider abstraction: does a second vendor really drop in?

The architectural claim this project makes is that adding a voice stack is a
translation table, not a rewrite. These tests are what make that claim checkable
rather than aspirational. They drive the Gemini-shaped mock and hand-written
OpenAI-shaped frames through the *same* normalization contract and assert the
harness sees the same event vocabulary from both.

The OpenAI adapter has no credentials and has never touched the wire. What is
tested here is precisely the part that does not need one -- :func:`translate`,
a pure function -- and the tests are labelled so nobody mistakes that for an
integration test.
"""

from __future__ import annotations

import base64

import pytest

from voiceval.audio import fixtures as fx
from voiceval.audio.pcm import PCM
import voiceval.providers  # noqa: F401  (registers every adapter)
from voiceval.providers import openai_realtime as oai
from voiceval.providers.base import (
    EventKind,
    ProviderUnavailable,
    SessionConfig,
    ToolSpec,
    TurnDetection,
    VirtualClock,
    get_provider,
    registered_providers,
)
from voiceval.providers.mock import MockScript, MockToolCall, MockTurn, MockVoiceProvider


def cfg(**kw) -> SessionConfig:
    kw.setdefault("system_instruction", "be helpful")
    return SessionConfig(**kw)


async def drain(session, limit: int = 500):
    out = []
    for _ in range(limit):
        got = False
        async for ev in session.events():
            out.append(ev)
            got = True
            if ev.kind in (
                EventKind.AGENT_TURN_COMPLETE,
                EventKind.INTERRUPTED,
                EventKind.TOOL_CALL,
                EventKind.ERROR,
            ):
                return out
        if not got:
            return out
    return out


class TestRegistry:
    def test_all_three_providers_are_registered(self):
        names = registered_providers()
        assert {"mock", "gemini_live", "openai_realtime"} <= set(names)

    def test_unknown_provider_names_the_alternatives(self):
        with pytest.raises(KeyError, match="registered"):
            get_provider("nope")

    def test_wire_verification_status_is_declared_not_assumed(self):
        assert get_provider("gemini_live").wire_verified is True
        assert get_provider("openai_realtime").wire_verified is True


class TestMockSession:
    async def test_a_turn_produces_audio_and_completes(self):
        p = MockVoiceProvider(MockScript([MockTurn(text="hello", speech_duration_s=1.0)]))
        s = await p.connect(cfg(), VirtualClock())
        await s.send_audio(fx.speech_like(1.0), ground_truth_text="hi")
        await s.commit_turn()
        evs = await drain(s)
        kinds = [e.kind for e in evs]
        assert EventKind.AGENT_AUDIO in kinds
        assert kinds[-1] == EventKind.AGENT_TURN_COMPLETE

    async def test_scripted_response_delay_is_reproduced_exactly(self):
        p = MockVoiceProvider(
            MockScript([MockTurn(response_delay_s=0.62, ttfa_after_start_s=0.18)])
        )
        clock = VirtualClock()
        s = await p.connect(cfg(), clock)
        await s.send_audio(fx.speech_like(1.0))
        t_commit = clock.now()
        await s.commit_turn()
        evs = await drain(s)
        first_audio = next(e for e in evs if e.kind == EventKind.AGENT_AUDIO)
        assert first_audio.t - t_commit == pytest.approx(0.80, abs=1e-6)

    async def test_tool_call_round_trip_gates_the_speech(self):
        """The agent must not start talking before its tool result comes back."""
        p = MockVoiceProvider(
            MockScript(
                [MockTurn(tool_calls=[MockToolCall("get_account", {"account_id": "a"})],
                          post_tool_delay_s=0.34)]
            )
        )
        clock = VirtualClock()
        s = await p.connect(cfg(tools=(ToolSpec("get_account", "", {}),)), clock)
        await s.send_audio(fx.speech_like(1.0))
        await s.commit_turn()
        evs = await drain(s)
        assert evs[-1].kind == EventKind.TOOL_CALL
        assert not [e for e in evs if e.kind == EventKind.AGENT_AUDIO]

        clock.advance(0.25)  # the harness executes the tool
        await s.send_tool_result(evs[-1].call_id, "get_account", {"output": "ok"})
        more = await drain(s)
        audio = [e for e in more if e.kind == EventKind.AGENT_AUDIO]
        assert audio, "speech should follow the tool result"
        # turn start 0.62 + tool delay 0.15 = requested 0.77; harness took
        # 0.25 to run it; post-tool delay 0.34 -> first audio at 1.36.
        assert audio[0].t == pytest.approx(1.36, abs=1e-6)

    async def test_unknown_tool_result_id_is_rejected(self):
        p = MockVoiceProvider(MockScript([MockTurn()]))
        s = await p.connect(cfg(), VirtualClock())
        with pytest.raises(KeyError):
            await s.send_tool_result("bogus", "x", {})

    async def test_ground_truth_text_comes_back_as_the_asr_transcript(self):
        p = MockVoiceProvider(MockScript([MockTurn()]))
        s = await p.connect(cfg(), VirtualClock())
        await s.send_audio(fx.speech_like(1.0), ground_truth_text="my account is acct one")
        await s.commit_turn()
        evs = await drain(s)
        t = next(e for e in evs if e.kind == EventKind.CALLER_TRANSCRIPT)
        assert t.text == "my account is acct one" and t.is_final

    async def test_asr_can_be_scripted_to_mishear(self):
        p = MockVoiceProvider(MockScript([MockTurn(asr_text="my account is at one")]))
        s = await p.connect(cfg(), VirtualClock())
        await s.send_audio(fx.speech_like(1.0), ground_truth_text="my account is acct one")
        await s.commit_turn()
        evs = await drain(s)
        assert next(e for e in evs if e.kind == EventKind.CALLER_TRANSCRIPT).text == (
            "my account is at one"
        )


class TestMockBargeIn:
    async def test_caller_audio_mid_utterance_truncates_the_agent(self):
        p = MockVoiceProvider(
            MockScript([MockTurn(speech_duration_s=4.0, yield_latency_s=0.18)])
        )
        clock = VirtualClock()
        s = await p.connect(cfg(), clock)
        await s.send_audio(fx.speech_like(1.0))
        await s.commit_turn()

        # Pull events until the agent is a second into speaking.
        first_audio_t = None
        async for ev in s.events():
            if ev.kind == EventKind.AGENT_AUDIO:
                first_audio_t = ev.t
                break
        assert first_audio_t is not None
        clock.set(first_audio_t + 1.0)
        await s.send_audio(fx.speech_like(0.5))

        rest = await drain(s)
        interrupt = [e for e in rest if e.kind == EventKind.INTERRUPTED]
        assert interrupt, "server-side barge-in should signal an interruption"
        assert interrupt[0].t == pytest.approx(first_audio_t + 1.0 + 0.18, abs=1e-6)

    async def test_provider_without_barge_in_keeps_talking(self):
        """The near-miss: a provider that does not implement barge-in."""
        p = MockVoiceProvider(
            MockScript([MockTurn(speech_duration_s=4.0)]), server_barge_in=False
        )
        clock = VirtualClock()
        s = await p.connect(cfg(), clock)
        await s.send_audio(fx.speech_like(1.0))
        await s.commit_turn()
        async for ev in s.events():
            if ev.kind == EventKind.AGENT_AUDIO:
                clock.set(ev.t + 1.0)
                break
        await s.send_audio(fx.speech_like(0.5))
        rest = await drain(s)
        assert not [e for e in rest if e.kind == EventKind.INTERRUPTED]

    async def test_capabilities_declare_the_difference(self):
        assert MockVoiceProvider(server_barge_in=False).capabilities().server_barge_in is False
        assert MockVoiceProvider().capabilities().server_barge_in is True


class TestOpenAITranslation:
    """Pure translation only. Nothing here has touched the live service."""

    def t(self, frame):
        return oai.translate(frame, t=1.5, seq=7)

    def test_session_created_opens_the_session(self):
        assert self.t({"type": "session.created"})[0].kind == EventKind.SESSION_OPENED

    def test_response_created_is_a_turn_start(self):
        evs = self.t({"type": "response.created"})
        assert evs[0].kind == EventKind.AGENT_TURN_STARTED
        assert evs[0].t == 1.5 and evs[0].seq == 7

    @pytest.mark.parametrize(
        "kind", ["response.output_audio.delta", "response.audio.delta"]
    )
    def test_both_audio_delta_names_decode_to_pcm(self, kind):
        raw = PCM.silence(0.02, 24000).data
        evs = self.t({"type": kind, "delta": base64.b64encode(raw).decode()})
        assert evs[0].kind == EventKind.AGENT_AUDIO
        assert evs[0].audio.data == raw
        assert evs[0].audio.sample_rate_hz == 24000

    def test_empty_audio_delta_produces_nothing(self):
        assert self.t({"type": "response.output_audio.delta", "delta": ""}) == []

    def test_caller_transcription_completed_is_final(self):
        evs = self.t(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "my email is priya at northwind",
            }
        )
        assert evs[0].kind == EventKind.CALLER_TRANSCRIPT and evs[0].is_final

    def test_function_call_arguments_are_parsed_from_the_json_string(self):
        evs = self.t(
            {
                "type": "response.function_call_arguments.done",
                "call_id": "call_abc",
                "name": "issue_account_credit",
                "arguments": '{"account_id": "acct_1042", "amount_cents": 30000}',
            }
        )
        assert evs[0].kind == EventKind.TOOL_CALL
        assert evs[0].call_id == "call_abc"
        assert evs[0].tool_args["amount_cents"] == 30000

    def test_malformed_arguments_do_not_crash_the_stream(self):
        evs = self.t(
            {
                "type": "response.function_call_arguments.done",
                "call_id": "c",
                "name": "x",
                "arguments": "{not json",
            }
        )
        assert evs[0].kind == EventKind.TOOL_CALL and evs[0].tool_args == {}

    def test_response_done_completes_the_turn(self):
        evs = self.t({"type": "response.done", "response": {"status": "completed"}})
        assert evs[0].kind == EventKind.AGENT_TURN_COMPLETE

    def test_cancelled_response_is_an_interruption_not_a_completion(self):
        """The barge-in signal has to survive the vendor difference.

        Gemini says `interrupted`; OpenAI reports a response that finished with
        status `cancelled`. Both must arrive as INTERRUPTED or the barge-in
        metric would silently read zero for one of the two providers.
        """
        evs = self.t({"type": "response.done", "response": {"status": "cancelled"}})
        assert evs[0].kind == EventKind.INTERRUPTED

    def test_error_frames_carry_their_message(self):
        evs = self.t({"type": "error", "error": {"message": "rate limited"}})
        assert evs[0].kind == EventKind.ERROR and "rate limited" in evs[0].message

    def test_unknown_frames_are_ignored_rather_than_fatal(self):
        assert self.t({"type": "response.output_item.added"}) == []
        assert self.t({}) == []


class TestOpenAISessionFrame:
    """The GA `/v1/realtime` session shape.

    These assertions exist because this is the one part of the seam the live
    service invalidated. The adapter was originally written against the Realtime
    *Beta* shape; that API is switched off and now closes the socket with
    `beta_api_shape_disabled`. The server-to-client translation table needed no
    changes at all — only this frame did — so it is pinned here.
    """

    def test_frame_declares_the_ga_realtime_session_type(self):
        frame = oai.OpenAIRealtimeProvider().session_update_frame(cfg())
        assert frame["type"] == "session.update"
        assert frame["session"]["type"] == "realtime"

    def test_audio_formats_are_nested_objects_not_flat_strings(self):
        """The Beta shape used `input_audio_format: "pcm16"`. GA does not."""
        sess = oai.OpenAIRealtimeProvider().session_update_frame(cfg())["session"]
        assert "input_audio_format" not in sess and "output_audio_format" not in sess
        assert sess["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
        assert sess["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}

    def test_output_modalities_replaced_modalities(self):
        sess = oai.OpenAIRealtimeProvider().session_update_frame(cfg())["session"]
        assert sess["output_modalities"] == ["audio"]
        assert "modalities" not in sess

    def test_voice_and_turn_detection_live_under_audio(self):
        sess = oai.OpenAIRealtimeProvider().session_update_frame(
            cfg(voice="alloy")
        )["session"]
        assert sess["audio"]["output"]["voice"] == "alloy"
        assert "voice" not in sess
        assert "turn_detection" not in sess

    def test_tools_are_translated_into_the_vendor_shape(self):
        frame = oai.OpenAIRealtimeProvider().session_update_frame(
            cfg(tools=(ToolSpec("get_account", "look up an account",
                                {"type": "object", "properties": {}}),))
        )
        tool = frame["session"]["tools"][0]
        assert tool["type"] == "function" and tool["name"] == "get_account"
        assert frame["session"]["tool_choice"] == "auto"

    def test_client_commit_mode_disables_server_vad(self):
        sess = oai.OpenAIRealtimeProvider().session_update_frame(
            cfg(turn_detection=TurnDetection.CLIENT_COMMIT)
        )["session"]
        assert sess["audio"]["input"]["turn_detection"] is None

    def test_server_vad_is_the_default(self):
        sess = oai.OpenAIRealtimeProvider().session_update_frame(cfg())["session"]
        assert sess["audio"]["input"]["turn_detection"] == {"type": "server_vad"}

    def test_input_transcription_is_requested_when_asked_for(self):
        sess = oai.OpenAIRealtimeProvider().session_update_frame(cfg())["session"]
        assert sess["audio"]["input"]["transcription"]["model"] == "whisper-1"
        off = oai.OpenAIRealtimeProvider().session_update_frame(
            cfg(request_input_transcript=False)
        )["session"]
        assert "transcription" not in off["audio"]["input"]

    async def test_connect_refuses_without_a_key_instead_of_pretending(self):
        """An explicit empty key must not fall through to the ambient one."""
        p = oai.OpenAIRealtimeProvider(api_key="")
        with pytest.raises(ProviderUnavailable, match="OPENAI_API_KEY"):
            await p.connect(cfg())

    def test_capabilities_advertise_the_turn_start_frame_gemini_lacks(self):
        from voiceval.providers.gemini_live import GeminiLiveProvider

        assert oai.OpenAIRealtimeProvider().capabilities().emits_turn_start is True
        assert GeminiLiveProvider().capabilities().emits_turn_start is False


class TestOpenAILiveEventNames:
    """Event names confirmed emitted by the live GA service.

    Every one of these was in `translate()` before any key existed, written from
    published docs and tested only against fixtures I wrote. The live run
    changed none of them. That is the half of the seam claim that held, and
    pinning the names here stops a future edit from quietly breaking it.
    """

    OBSERVED = [
        "response.created",
        "response.output_audio.delta",
        "response.output_audio_transcript.delta",
        "response.output_audio_transcript.done",
        "response.function_call_arguments.done",
        "response.done",
        "conversation.item.input_audio_transcription.completed",
        "conversation.item.input_audio_transcription.delta",
    ]

    @pytest.mark.parametrize("kind", OBSERVED)
    def test_every_observed_event_type_is_handled(self, kind):
        import base64

        frame = {"type": kind,
                 "delta": base64.b64encode(b"\x00\x00" * 160).decode(),
                 "transcript": "x", "arguments": "{}",
                 "call_id": "c", "name": "n", "response": {"status": "completed"}}
        evs = oai.translate(frame, t=1.0, seq=1)
        assert evs, f"{kind} produced no normalized event"

    def test_ga_only_bookkeeping_frames_are_ignored_not_fatal(self):
        """GA adds frames the harness has no meaning for; they must be inert."""
        for kind in ("conversation.item.added", "conversation.item.done",
                     "response.output_item.added", "response.content_part.added",
                     "rate_limits.updated", "input_audio_buffer.committed"):
            assert oai.translate({"type": kind}, t=1.0, seq=1) == []
