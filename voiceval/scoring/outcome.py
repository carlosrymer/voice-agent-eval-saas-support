"""Outcome: did the business goal actually happen?

No model is involved. Each task carries environment assertions -- "the account
has exactly $300 of credit", "no escalation was raised" -- and tau2's
:class:`EnvironmentEvaluator` replays the call's actions into a fresh
environment and checks them. The article calls this a "business-system check",
and it is the strongest kind of eval signal available here: the answer does not
depend on anybody's opinion, including mine.

Running it through tau2's evaluator rather than my own assertion loop is what
makes the voice arm's Outcome score directly comparable with the published text
baseline for the same 16 tasks. Same tasks, same assertions, same evaluator,
different channel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import saas_support  # noqa: F401  isort:skip

from tau2.evaluator.evaluator_env import EnvironmentEvaluator  # noqa: E402

from voiceval.domain import build_trajectory, env_constructor, task_by_id  # noqa: E402
from voiceval.metrics.timeline import CallRecord  # noqa: E402


@dataclass
class OutcomeScore:
    task_id: str
    #: 1.0 when every environment assertion for this task holds.
    reward: float | None
    #: Stricter secondary signal: the whole database matches a gold replay.
    db_match: bool | None
    assertions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.reward == 1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def score_outcome(record: CallRecord) -> OutcomeScore:
    task = task_by_id(record.task_id)
    trajectory = build_trajectory(record)
    try:
        info = EnvironmentEvaluator.calculate_reward(
            environment_constructor=env_constructor,
            task=task,
            full_trajectory=trajectory,
        )
    except Exception as exc:
        # A harness failure must not be scored as a task failure -- that would
        # quietly manufacture voice-channel degradation out of my own bugs.
        return OutcomeScore(
            task_id=record.task_id,
            reward=None,
            db_match=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    checks: list[dict[str, Any]] = []
    for c in getattr(info, "env_assertions", None) or []:
        checks.append(
            {
                "func_name": getattr(getattr(c, "env_assertion", None), "func_name", None),
                "met": bool(getattr(c, "met", False)),
                "reward": getattr(c, "reward", None),
            }
        )
    db_check = getattr(info, "db_check", None)
    return OutcomeScore(
        task_id=record.task_id,
        reward=float(info.reward),
        db_match=bool(getattr(db_check, "db_match", False)) if db_check else None,
        assertions=checks,
    )
