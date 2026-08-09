"""Environment wiring for the Loopline SaaS support domain."""

from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.utils import load_file

from saas_support.data_model import SaasDB
from saas_support.tools import SaasTools
from saas_support.user_data_model import SaasUserDB
from saas_support.user_tools import SaasUserTools
from saas_support.utils import (
    DOMAIN_NAME,
    SAAS_DB_PATH,
    SAAS_POLICY_PATH,
    SAAS_TASK_SET_PATH,
    SAAS_USER_DB_PATH,
    domain_verification_token,
)


class SaasEnvironment(Environment):
    """Dual-control environment.

    `sync_tools` is the bridge between the two halves: it is how an action the
    *customer* takes in their own workspace becomes visible to the support
    agent's console. Domain verification is the clearest case — the agent can
    read whether a domain is verified, but only the customer can publish the
    TXT record that makes it so.
    """

    tools: SaasTools
    user_tools: SaasUserTools

    def __init__(
        self,
        domain_name: str,
        policy: str,
        tools: SaasTools,
        user_tools: SaasUserTools,
    ):
        super().__init__(domain_name, policy, tools, user_tools)

    def make_tool_call(self, tool_name: str, requestor: str = "assistant", **kwargs):
        """Call a tool and then re-sync the two halves of the environment.

        The base `Environment.make_tool_call` documents "this does not call
        sync_tools". That creates an asymmetry that matters for a dual-control
        domain: `EnvironmentEvaluator` builds the *gold* environment by calling
        `make_tool_call` in a loop (never syncing), while the *predicted*
        environment is built with `set_state`, which finishes with a
        `sync_tools()`. So any task whose correct solution depends on a
        cross-environment effect — here, the customer publishing a DNS record
        flipping `domain_verified` on the agent's side — would show a gold DB
        that never reflects the sync and a predicted DB that does, and could
        never match on the DB hash no matter what the agent did.

        Syncing here makes the two paths symmetric. `sync_tools` is idempotent,
        so the extra call from `get_response` is harmless.
        """
        result = super().make_tool_call(tool_name, requestor=requestor, **kwargs)
        self.sync_tools()
        return result

    def sync_tools(self):
        if self.tools is None or self.user_tools is None:
            return
        workspace = self.user_tools.db.workspace
        account = self.tools.db.accounts.get(workspace.account_id)
        if account is None:
            return
        # Publishing the right TXT record verifies the sending domain.
        expected = domain_verification_token(account.account_id)
        if expected in workspace.dns_txt_records:
            account.domain_verified = True


def get_environment(
    db: Optional[SaasDB] = None,
    user_db: Optional[SaasUserDB] = None,
    solo_mode: bool = False,
) -> SaasEnvironment:
    if db is None:
        db = SaasDB.load(SAAS_DB_PATH)
    if user_db is None:
        user_db = SaasUserDB.load(SAAS_USER_DB_PATH)
    policy = load_file(SAAS_POLICY_PATH)
    env = SaasEnvironment(
        domain_name=DOMAIN_NAME,
        policy=policy,
        tools=SaasTools(db),
        user_tools=SaasUserTools(user_db),
    )
    if solo_mode:
        env.set_solo_mode(True)
    return env


def load_tasks(path) -> list[Task]:
    tasks = load_file(path)
    if isinstance(tasks, dict) and "tasks" in tasks:
        tasks = tasks["tasks"]
    return [Task.model_validate(task) for task in tasks]


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = load_tasks(SAAS_TASK_SET_PATH)
    if task_split_name is None:
        return tasks
    splits = get_tasks_split()
    if task_split_name not in splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. "
            f"Valid splits are: {list(splits)}"
        )
    return [task for task in tasks if task.id in splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(SAAS_TASK_SET_PATH).parent / f"split_{Path(SAAS_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
