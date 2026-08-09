"""Paths and fixed clock for the Loopline SaaS support domain."""

import os
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# tau2 resolves its own DATA_DIR relative to a source checkout. Installing it
# from git means that directory is absent, so point tau2 at the small vendored
# copy of the framework files it needs (the user-simulator guidelines). This
# must be set before anything under `tau2` is imported.
os.environ.setdefault("TAU2_DATA_DIR", str(PROJECT_ROOT / "tau2_data"))

# This domain's own data lives in the project, not inside the tau2 package.
DOMAIN_DATA_DIR = PROJECT_ROOT / "domain_data"

SAAS_DB_PATH = DOMAIN_DATA_DIR / "db.json"
SAAS_USER_DB_PATH = DOMAIN_DATA_DIR / "user_db.json"
SAAS_POLICY_PATH = DOMAIN_DATA_DIR / "policy.md"
SAAS_TASK_SET_PATH = DOMAIN_DATA_DIR / "tasks.json"

DOMAIN_NAME = "saas_support"

# The credit ceiling an agent may approve without escalation, in cents.
CREDIT_ESCALATION_THRESHOLD_CENTS = 50_000  # $500.00


def get_now() -> datetime:
    """Frozen clock so tasks and billing cycles are reproducible."""
    return datetime(2026, 3, 14, 10, 30, 0)


def get_today() -> date:
    return get_now().date()


def domain_verification_token(account_id: str) -> str:
    """The DNS TXT value that proves control of an account's sending domain.

    Deterministic on purpose: a random token would make the gold-vs-predicted
    DB hash comparison non-reproducible.
    """
    return f"loopline-verify={account_id}"


def api_key_for(account_id: str, rotations: int) -> str:
    """Deterministic API key so rotation is reproducible across replays."""
    return f"lk_live_{account_id}_{rotations:02d}"
