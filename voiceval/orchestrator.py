"""Drives one voice call and records everything needed to score it.

## The playout buffer, and why arrival time is not playback time

A realtime API streams audio faster than it is spoken -- Gemini Live will hand
over three seconds of speech in a few hundred milliseconds. If the agent track
were assembled by placing each chunk at the moment it arrived, the reconstructed
recording would be several times shorter than the utterance the caller actually
heard, every barge-in would look like it happened after the agent had finished,
and overlap would measure as zero.

So the agent track is reconstructed the way a real client plays it: the first
chunk of a turn starts playing when it arrives, subsequent chunks play back to
back at the sample rate, and if the stream stalls past the end of what is
buffered the playhead jumps forward to the next arrival -- an underrun, which is
a gap the caller genuinely hears. On an interruption the buffered remainder is
discarded, because that is what a client does and what the caller experiences.

Everything the Experience metrics read is therefore a reconstruction of the
caller's ear, not of the socket. That is a modelling choice and it is the single
most consequential one in this harness, which is why it is described here and in
the README rather than left implicit in a loop.

## Barge-in

Interruptions are scripted (see :class:`BargeInPlan`). At the scheduled offset
into a chosen agent turn, a background task pushes caller audio while the agent
is still speaking. Gemini Live detects it server-side and reports
``interrupted``. Attempts that arrive after the agent already stopped are
recorded as misses so the barge-in sample size is always honest.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from tau2.data_model.tasks import Task

from voiceval.audio.pcm import PCM, place, resample
from voiceval.caller.simulator import BargeInPlan, CallerSimulator
from voiceval.domain import agent_tool_specs, execute_agent_tool, new_environment, voice_system_prompt
from voiceval.llm import GeminiClient
from voiceval.metrics.timeline import AgentUtterance, CallerUtterance, CallRecord, ToolExecution
from voiceval.providers.base import (
    EventKind,
    PausableWallClock,
    ServerEvent,
    SessionConfig,
    VoiceProvider,
)
from voiceval.tts import TTSBackend, TTSQuotaExhausted

CALLER_RATE = 16000
#: Length of each comfort-silence frame used to hold the line open.
LINE_FRAME_S = 0.02


@dataclass
class CallConfig:
    max_turns: int = 14
    caller_voice: str = "Kore"
    agent_voice: str = "Puck"
    #: Hard stop on wall-clock seconds for one call.
    max_call_s: float = 300.0
    #: How long to wait for the agent's *first* frame of a turn. Covers server
    #: end-of-speech detection plus model time-to-first-token.
    first_frame_s: float = 15.0
    #: How long to keep draining events after the provider goes quiet mid-turn.
    turn_idle_s: float = 3.0
    #: Cap on how long to wait for the agent's buffered speech to finish
    #: playing before the caller is allowed to talk again.
    max_playout_wait_s: float = 45.0
    #: Silence appended after each caller utterance.
    #:
    #: Not cosmetic. With server-side turn detection the agent only starts
    #: responding once it has heard enough silence to decide the caller has
    #: stopped, so an utterance that ends on its last syllable is never
    #: endpointed and the agent never replies at all -- the first live run of
    #: this harness produced eight caller turns and zero agent turns for
    #: exactly this reason. The pause is also what a real caller leaves, and
    #: the endpointing delay it triggers is a genuine part of the latency a
    #: caller perceives, so it belongs inside the measurement rather than
    #: being engineered away.
    trailing_silence_s: float = 0.8
    barge_in: BargeInPlan = field(default_factory=BargeInPlan)
    caller_model: str = "gemini-3.6-flash"
    caller_temperature: float = 1.0
    trial: int = 0


class Playout:
    """Reconstructs the audio a caller actually hears from arriving chunks."""

    def __init__(self, rate: int):
        self.rate = rate
        self.chunks: list[tuple[float, PCM]] = []
        self._playhead: float | None = None
        self.underruns: int = 0

    def add(self, arrival_t: float, pcm: PCM) -> tuple[float, float]:
        if self._playhead is None or arrival_t > self._playhead + 1e-6:
            if self._playhead is not None:
                self.underruns += 1
            start = arrival_t
        else:
            start = self._playhead
        self.chunks.append((start, pcm))
        self._playhead = start + pcm.duration_s
        return start, self._playhead

    def interrupt(self, at_t: float) -> None:
        """Drop everything that would have played after ``at_t``."""
        kept: list[tuple[float, PCM]] = []
        for start, pcm in self.chunks:
            end = start + pcm.duration_s
            if start >= at_t:
                continue
            if end > at_t:
                pcm = pcm.slice_s(0.0, at_t - start)
            kept.append((start, pcm))
        self.chunks = kept
        self._playhead = None

    def end_turn(self) -> None:
        self._playhead = None

    @property
    def last_end(self) -> float:
        return max((s + p.duration_s for s, p in self.chunks), default=0.0)

    def render(self, total_s: float) -> PCM:
        track = PCM.silence(max(total_s, self.last_end), self.rate)
        for start, pcm in self.chunks:
            if pcm.n_samples:
                track = place(track, pcm, start)
        return track


async def run_call(
    task: Task,
    provider: VoiceProvider,
    tts: TTSBackend,
    client: GeminiClient,
    config: CallConfig | None = None,
    caller_factory=None,
) -> CallRecord:
    """Run one call. ``caller_factory(task, env, client, cfg)`` overrides the
    default LLM-driven caller, which is how the offline end-to-end test drives
    the entire pipeline with no API key and no spend."""
    cfg = config or CallConfig()
    env = new_environment(task)
    tools = agent_tool_specs(env)
    caps = provider.capabilities()

    session_cfg = SessionConfig(
        system_instruction=voice_system_prompt(),
        tools=tuple(tools),
        voice=cfg.agent_voice,
        temperature=1.0,
    )
    clock = PausableWallClock()
    session = await provider.connect(session_cfg, clock)

    caller = (
        caller_factory(task, env, client, cfg)
        if caller_factory is not None
        else CallerSimulator(task, env, client, model=cfg.caller_model,
                             temperature=cfg.caller_temperature)
    )

    call_id = f"{task.id}__t{cfg.trial}__{uuid.uuid4().hex[:6]}"
    record = CallRecord(
        call_id=call_id,
        task_id=task.id,
        trial=cfg.trial,
        provider=provider.name,
        model=getattr(provider, "model", provider.name),
        synthetic=provider.name == "mock",
        capabilities=caps.to_dict(),
    )
    playout = Playout(caps.output_sample_rate_hz)
    caller_chunks: list[tuple[float, PCM]] = []
    barge_attempts = 0
    started = time.monotonic()

    async def speak(text: str, *, barge_in: bool = False) -> None:
        """Say one line, and keep the recorded timeline honest while doing it.

        The caller occupies real time for as long as the utterance lasts, but the
        audio is *transmitted* as a single burst bracketed by explicit activity
        markers rather than streamed at speaking pace. Streaming it turned out to
        give the server room to begin a response part-way through the utterance
        and then cancel it -- taking the agent's in-flight tool call with it --
        which made every tool-using task fail from the second turn onwards.

        The two cases differ in ordering, and the difference matters:

        * A normal turn sleeps for the utterance duration and *then* transmits,
          so the server cannot possibly answer before the caller has finished. A
          real streaming server could pipeline recognition and start earlier, so
          the latency measured this way is **pessimistic**, which is the safer
          direction to be wrong in.
        * A barge-in transmits immediately, because the whole point is that the
          interruption lands while the agent is still talking. The server
          therefore hears the entire interruption at its onset rather than over
          its duration, so it can react slightly sooner than a real one would.
        """
        clock.pause()
        try:
            speech = resample(tts.synthesize(text, cfg.caller_voice), CALLER_RATE)
        finally:
            clock.resume()
        await utter(text, speech, barge_in=barge_in)

    async def utter(text: str, speech: PCM, *, barge_in: bool = False) -> None:
        """Transmit an already-synthesised utterance.

        Split out from `speak` because the barge-in path must not synthesise
        anything: it runs *during* a live agent turn, and synthesis pauses the
        session clock. Pausing mid-turn froze the turn's own idle deadline and
        stalled the conversation for as long as the TTS round trip took, while
        the agent's audio kept arriving and being stamped with a clock that was
        not moving. The interruption is therefore rendered once, before the call
        starts, and only transmitted here.
        """
        t0 = clock.now()
        dur = speech.duration_s
        # The recorded track holds the speech only; the trailing pause is silence
        # either way, so it moves no VAD boundary and `end_t` stays the moment
        # the caller actually stopped talking.
        caller_chunks.append((t0, speech))
        record.caller_utterances.append(
            CallerUtterance(
                index=len(record.caller_utterances),
                text=text,
                start_t=t0,
                end_t=t0 + dur,
                is_barge_in=barge_in,
            )
        )
        padded = speech + PCM.silence(cfg.trailing_silence_s, CALLER_RATE)
        if barge_in:
            caller_speaking.set()
            await session.send_audio(padded, ground_truth_text=text)
            await session.commit_turn()
            remaining = (t0 + dur) - clock.now()
            if remaining > 0:
                await asyncio.sleep(remaining)
        else:
            await asyncio.sleep(dur)
            await session.send_audio(padded, ground_truth_text=text)
            await session.commit_turn()

    #: Set while a scripted barge-in utterance is on the wire, so a tool-call
    #: cancellation can be attributed to the caller rather than to the server.
    caller_speaking = asyncio.Event()

    barge_pcm: PCM | None = None
    if cfg.barge_in.turns:
        clock.pause()
        try:
            barge_pcm = resample(
                tts.synthesize(cfg.barge_in.utterance, cfg.caller_voice), CALLER_RATE
            )
        except Exception as exc:
            record.errors.append(f"barge-in synthesis failed: {exc}")
        finally:
            clock.resume()

    try:
        for turn_index in range(cfg.max_turns):
            if time.monotonic() - started > cfg.max_call_s:
                record.ended_reason = "max_call_seconds"
                break

            clock.pause()
            try:
                action = caller.next_action()
            finally:
                clock.resume()
            if action.kind == "end":
                record.ended_reason = _reason(action.terminator)
                break
            try:
                await speak(action.text)
            except TTSQuotaExhausted as exc:
                # Out of speech-synthesis budget. The call stops here and is
                # marked as a harness failure so scoring drops it, rather than
                # letting a truncated conversation be counted as the agent
                # failing the task.
                record.errors.append(f"TTS quota exhausted: {exc}")
                record.ended_reason = "tts_quota_exhausted"
                break
            if action.terminator:
                record.ended_reason = _reason(action.terminator)

            speaking = asyncio.Event()
            barge_task: asyncio.Task | None = None
            if cfg.barge_in.fires_on(turn_index) and barge_pcm is not None:
                barge_attempts += 1
                barge_task = asyncio.create_task(
                    _delayed_barge_in(cfg.barge_in, utter, barge_pcm, speaking)
                )

            line_stop = asyncio.Event()
            line_task = (
                asyncio.create_task(
                    _hold_line_open(session, caps.input_sample_rate_hz, line_stop)
                )
                if caps.server_turn_detection
                else None
            )
            try:
                agent_text, interrupted = await _consume_turn(
                    session, record, env, playout, clock, cfg, speaking, caller_speaking
                )
            finally:
                if line_task is not None:
                    line_stop.set()
                    line_task.cancel()
                    try:
                        await line_task
                    except (asyncio.CancelledError, Exception):
                        pass
            if barge_task is not None:
                barge_task.cancel()
                try:
                    await barge_task
                except (asyncio.CancelledError, Exception):
                    pass

            record.agent_utterances.append(
                AgentUtterance(
                    index=len(record.agent_utterances),
                    text=agent_text,
                    audio_start_t=_turn_audio_start(playout, record),
                    audio_end_t=playout.last_end or None,
                    turn_started_t=None,
                    completed_t=clock.now(),
                    interrupted=interrupted,
                )
            )
            # Let the agent finish being heard before the caller speaks again.
            # Audio arrives far faster than it is spoken, so without this the
            # caller starts talking while the agent is still mid-sentence --
            # which Gemini Live correctly treats as an interruption and which
            # cancelled the agent's in-flight tool calls on the first live run.
            remaining = playout.last_end - clock.now()
            if remaining > 0:
                await asyncio.sleep(min(remaining, cfg.max_playout_wait_s))
            playout.end_turn()
            caller.observe_agent(agent_text)
            if record.ended_reason != "completed":
                break
        else:
            record.ended_reason = "max_turns"
    except TTSQuotaExhausted as exc:
        record.errors.append(f"TTS quota exhausted: {exc}")
        record.ended_reason = "tts_quota_exhausted"
    except Exception as exc:
        record.errors.append(f"{type(exc).__name__}: {exc}")
        record.ended_reason = "error"
    finally:
        await session.close()

    total = max(clock.now(), playout.last_end)
    record.duration_s = total
    caller_track = PCM.silence(total, CALLER_RATE)
    for t0, pcm in caller_chunks:
        caller_track = place(caller_track, pcm, t0)
    record.caller_track = caller_track
    record.agent_track = playout.render(total)
    record.meta.update(
        {
            "playout_underruns": playout.underruns,
            "barge_in_plan": {
                "turns": list(cfg.barge_in.turns),
                "offset_s": cfg.barge_in.offset_s,
                "utterance": cfg.barge_in.utterance,
            },
            "barge_in_attempts": barge_attempts,
            "caller_model": cfg.caller_model,
            "caller_voice": cfg.caller_voice,
            "agent_voice": cfg.agent_voice,
            "tts_backend": tts.name,
            "caller_tool_calls": [
                {"name": n, "args": a} for n, a, _ in caller.tool_calls_made
            ],
            "realtime_pacing": getattr(provider, "realtime_pacing", None),
            "harness_paused_s": round(getattr(clock, "total_paused_s", 0.0), 3),
        }
    )
    record.meta.pop("_executed_tools", None)
    _attach_asr(record)
    record.meta["env_db_hash"] = _safe_hash(env)
    return record


async def _hold_line_open(session, rate: int, stop: asyncio.Event) -> None:
    """Keep streaming silence while waiting for the agent.

    A real phone line carries silence between utterances; this harness was
    instead sending a burst of speech and then nothing at all. Server-side
    end-of-speech detection needs to *hear* the silence to decide the caller has
    finished, so with a dead stream the agent sometimes never replied -- the
    failure was intermittent, because whether it endpointed depended on how much
    trailing silence happened to be inside the last buffer.

    Holding the line open also keeps the endpointing delay inside the measured
    end-of-turn latency, where it belongs: it is part of what a caller waits
    through. The alternative -- signalling end-of-audio explicitly -- is more
    reliable still, but it skips exactly that delay and would make every latency
    figure optimistic in a way that is invisible in the output.
    """
    frame = PCM.silence(LINE_FRAME_S, rate)
    try:
        while not stop.is_set():
            await session.send_audio(frame)
            await asyncio.sleep(LINE_FRAME_S)
    except asyncio.CancelledError:
        raise
    except Exception:
        return


async def _delayed_barge_in(
    plan: BargeInPlan, utter, pcm: PCM, speaking: asyncio.Event
) -> None:
    """Interrupt `offset_s` after the agent actually starts speaking.

    Timing from the start of the turn instead would fire the interruption while
    the model was still thinking, so it would land in silence and test nothing
    -- and the barge-in metrics would be computed over an empty sample while
    still reporting a yield rate.
    """
    await speaking.wait()
    await asyncio.sleep(plan.offset_s)
    await utter(plan.utterance, pcm, barge_in=True)


async def _consume_turn(
    session, record: CallRecord, env, playout: Playout, clock, cfg,
    speaking: asyncio.Event | None = None,
    caller_speaking: "asyncio.Event | None" = None,
):
    """Drain provider events for one agent turn. Returns (text, interrupted)."""
    # (time, text, is_final) for every transcript frame seen while draining this
    # turn, including ones that turn out to belong to the previous turn.
    heard: list[tuple[float, str, bool]] = []
    first_content_t: float | None = None
    interrupted = False
    idle_deadline = clock.now() + cfg.first_frame_s
    done = False
    # Whether this turn has produced anything of its own yet.
    #
    # Gemini Live flushes the *previous* turn's final transcript and its
    # `turnComplete` the instant new caller audio arrives. Only audio and tool
    # calls count as proof that a frame belongs to *this* turn: a transcript
    # frame does not, because the flushed one is a transcript. An earlier
    # version accepted transcripts as proof, so every turn ended immediately
    # wearing the previous turn's words -- which then made the caller talk over
    # the agent's real answer and cancelled its in-flight tool calls. Responses
    # here are always audio (`responseModalities: ["AUDIO"]`), so requiring
    # audio or a tool call loses nothing.
    content_seen = False
    #: Audio is the only proof the agent actually said something this turn.
    audio_seen = False
    #: Tool calls issued in this turn that have not yet been followed by speech.
    tools_pending = False
    caller_is_speaking = False
    #: (name, args) -> (finished_t, result) for replay de-duplication.
    executed_tools: dict = record.meta.setdefault("_executed_tools", {})

    while not done and clock.now() < idle_deadline:
        caller_is_speaking = bool(caller_speaking and caller_speaking.is_set())
        got_any = False
        async for ev in session.events():
            got_any = True
            seen_any = True
            record.events.append(_light(ev))
            idle_deadline = clock.now() + cfg.turn_idle_s

            if ev.kind == EventKind.AGENT_AUDIO and ev.audio is not None:
                if not content_seen:
                    content_seen, first_content_t = True, ev.t
                audio_seen = True
                tools_pending = False
                playout.add(ev.t, ev.audio)
                if speaking is not None and not speaking.is_set():
                    speaking.set()
            elif ev.kind == EventKind.AGENT_TRANSCRIPT:
                if ev.text:
                    heard.append((ev.t, ev.text, ev.is_final))
            elif ev.kind == EventKind.TOOL_CALL:
                if not content_seen:
                    content_seen, first_content_t = True, ev.t
                tools_pending = True
                await _run_tool(session, record, env, ev, clock, executed_tools)
                # Generating an answer from a tool result takes as long as a
                # fresh response, so the turn gets a fresh full budget rather
                # than the short mid-turn idle window.
                idle_deadline = clock.now() + cfg.first_frame_s
            elif ev.kind == EventKind.INTERRUPTED:
                if not content_seen:
                    continue
                is_tool_cancel = "tool call cancelled" in (ev.message or "")
                if is_tool_cancel and not caller_is_speaking:
                    # The server abandoned a function call it had issued, with
                    # nobody talking over it. That is a server-side retry, not a
                    # barge-in: counting it as one would both end the turn early
                    # and inflate the barge-in numbers with events no caller
                    # caused. It usually re-issues the call moments later.
                    record.meta["server_tool_retries"] = (
                        record.meta.get("server_tool_retries", 0) + 1
                    )
                    continue
                interrupted = True
                playout.interrupt(ev.t)
                done = True
                break
            elif ev.kind == EventKind.AGENT_TURN_COMPLETE:
                if not content_seen:
                    continue  # stale frame for the previous turn
                if tools_pending and not audio_seen:
                    # The turn "completed" having done nothing but call a tool.
                    # The answer built from that tool result is still coming, so
                    # keep draining rather than handing the floor back to the
                    # caller -- doing so made the caller talk over the agent's
                    # real answer and cancelled the next tool call, repeatedly.
                    continue
                done = True
                break
            elif ev.kind == EventKind.ERROR:
                record.errors.append(ev.message or "provider error")
                done = True
                break
            elif ev.kind == EventKind.SESSION_CLOSED:
                done = True
                break
        if not got_any:
            await asyncio.sleep(0.02)

    # Gemini Live streams `outputTranscription` word by word and only emits a
    # consolidated final when the turn closes -- and an interrupted turn, or one
    # this loop leaves on an idle timeout, never gets there. So the fragments are
    # accumulated, and the consolidated final is preferred when it arrives.
    #
    # Transcript frames that predate this turn's first audio or tool call are the
    # previous turn's flush and are dropped. The half-second of slack covers the
    # normal case where the first word of the transcript beats the first audio
    # packet by a few tens of milliseconds.
    cutoff = (first_content_t - 0.5) if first_content_t is not None else float("inf")
    mine = [(t, txt, fin) for (t, txt, fin) in heard if t >= cutoff]
    finals = [txt for (_, txt, fin) in mine if fin]
    fragments = [txt for (_, txt, fin) in mine if not fin]
    text = " ".join(finals).strip() if finals else "".join(fragments).strip()
    return text, interrupted


#: How long a cancelled tool call stays eligible for replay de-duplication.
TOOL_REPLAY_WINDOW_S = 45.0


async def _run_tool(session, record: CallRecord, env, ev: ServerEvent, clock,
                    executed: dict | None = None) -> None:
    """Execute one tool call, guarding against server-side replays.

    Gemini Live sometimes cancels a function call it has already issued and then
    re-issues an identical one. Both arrive as real tool calls, and executing
    both applies the *write twice* -- three `issue_account_credit` calls for one
    credit the caller asked for once. That is not the agent misbehaving, and
    letting it through would have been the worst kind of bug in this project: it
    fabricates a P2 "split the credit to stay under the threshold" policy
    violation out of a transport retry, and corrupts the environment the Outcome
    assertions are checked against.

    So an identical (name, arguments) call seen again within
    `TOOL_REPLAY_WINDOW_S` returns the first result instead of re-executing. The
    replay is counted in `meta["duplicate_tool_calls"]` and does not enter the
    action ledger, so the policy auditor sees the one call the agent actually
    intended. A genuine repeat by the agent much later still executes normally,
    which is what keeps real split-credit violations detectable.
    """
    key = (ev.tool_name or "", json.dumps(ev.tool_args or {}, sort_keys=True, default=str))
    now = clock.now()
    if executed is not None and key in executed:
        prev_t, prev_result = executed[key]
        if now - prev_t <= TOOL_REPLAY_WINDOW_S:
            record.meta["duplicate_tool_calls"] = (
                record.meta.get("duplicate_tool_calls", 0) + 1
            )
            await session.send_tool_result(
                ev.call_id or "", ev.tool_name or "", {"output": prev_result}
            )
            return

    t_start = clock.now()
    result = execute_agent_tool(env, ev.tool_name or "", ev.tool_args or {})
    t_end = clock.now()
    if executed is not None:
        executed[key] = (t_end, result.content)
    record.tool_executions.append(
        ToolExecution(
            call_id=ev.call_id or "",
            name=ev.tool_name or "",
            args=dict(ev.tool_args or {}),
            requestor="assistant",
            requested_t=ev.t,
            started_t=t_start,
            finished_t=t_end,
            ok=result.ok,
            result=result.content,
            error=result.error,
        )
    )
    await session.send_tool_result(
        ev.call_id or "", ev.tool_name or "", {"output": result.content}
    )


def _light(ev: ServerEvent) -> ServerEvent:
    """Drop audio payloads from the stored event log; the WAVs hold the audio."""
    if ev.audio is None:
        return ev
    from dataclasses import replace

    return replace(ev, audio=PCM(b"", ev.audio.sample_rate_hz), raw={
        **ev.raw, "audio_ms": round(ev.audio.duration_s * 1000, 3)
    })


def _turn_audio_start(playout: Playout, record: CallRecord) -> float | None:
    prior_end = max(
        [a.audio_end_t or 0.0 for a in record.agent_utterances] or [0.0]
    )
    starts = [s for s, _ in playout.chunks if s >= prior_end - 1e-9]
    return min(starts) if starts else None


def _attach_asr(record: CallRecord) -> None:
    """Pair each caller utterance with the provider's final transcript of it."""
    finals = [
        e for e in record.events
        if e.kind == EventKind.CALLER_TRANSCRIPT and e.is_final and e.text
    ]
    for u in record.caller_utterances:
        after = [e for e in finals if e.t >= u.end_t - 0.5]
        if after:
            best = min(after, key=lambda e: e.t)
            u.asr_text = best.text
            u.asr_final_t = best.t
            finals.remove(best)


def _reason(terminator: str | None) -> str:
    return {
        "###STOP###": "caller_done",
        "###TRANSFER###": "transferred",
        "###OUT-OF-SCOPE###": "out_of_scope",
    }.get(terminator or "", "completed")


def _safe_hash(env) -> str | None:
    try:
        return env.get_db_hash()
    except Exception:
        return None
