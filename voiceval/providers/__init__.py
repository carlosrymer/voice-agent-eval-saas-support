"""Voice provider adapters.

Importing this package registers every adapter, so `get_provider(name)` works
without the caller knowing which module defines what. Adapters are imported for
their registration side effect; `gemini_live` and `openai_realtime` both import
cleanly with no credentials present, and only fail at `connect()`.
"""

from voiceval.providers.base import (  # noqa: F401
    Clock,
    EventKind,
    PausableWallClock,
    ProviderCapabilities,
    ProviderUnavailable,
    ServerEvent,
    SessionConfig,
    ToolSpec,
    TurnDetection,
    VirtualClock,
    VoiceProvider,
    VoiceSession,
    WallClock,
    get_provider,
    register_provider,
    registered_providers,
)
from voiceval.providers.gemini_live import GeminiLiveProvider  # noqa: F401
from voiceval.providers.mock import MockVoiceProvider  # noqa: F401
from voiceval.providers.openai_realtime import OpenAIRealtimeProvider  # noqa: F401
