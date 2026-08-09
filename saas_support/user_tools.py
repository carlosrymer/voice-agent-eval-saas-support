"""User-side tools: what the customer can do in their own workspace.

These are the actions no support agent can perform on the customer's behalf —
reading their inbox, publishing a DNS record, rotating their own API key,
flipping a toggle inside their workspace. Tasks are authored so that the agent
must recognise it needs something from this side and ask for it explicitly.
"""

from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

from saas_support.user_data_model import Email, SaasUserDB
from saas_support.utils import api_key_for


class SaasUserTools(ToolKitBase):
    """Tools available to the customer (the simulated user)."""

    db: SaasUserDB

    def __init__(self, db: SaasUserDB) -> None:
        super().__init__(db)

    # ---------- read tools ----------

    @is_tool(ToolType.READ)
    def check_inbox(self) -> list[Email]:
        """
        Open your email inbox and read what is in it.

        Returns:
            Every email currently in your inbox.
        """
        return list(self.db.inbox.values())

    @is_tool(ToolType.READ)
    def get_api_key(self) -> str:
        """
        Show your workspace's current API key. This is a secret credential.

        Returns:
            Your current API key.
        """
        return api_key_for(
            self.db.workspace.account_id, self.db.workspace.api_key_rotations
        )

    @is_tool(ToolType.READ)
    def list_dns_txt_records(self) -> list[str]:
        """
        List the DNS TXT records currently published on your sending domain.

        Returns:
            The TXT record values currently published.
        """
        return list(self.db.workspace.dns_txt_records)

    @is_tool(ToolType.READ)
    def get_workspace_settings(self) -> dict:
        """
        Show the current toggles in your Loopline workspace settings page.

        Returns:
            A mapping of setting name to whether it is enabled.
        """
        return dict(self.db.workspace.settings)

    # ---------- write tools ----------

    @is_tool(ToolType.WRITE)
    def add_dns_txt_record(self, value: str) -> str:
        """
        Publish a DNS TXT record on your sending domain. You will need the exact
        value from the support agent.

        Args:
            value: The exact TXT record value to publish.

        Returns:
            Confirmation that the record was published.
        """
        value = value.strip()
        if value not in self.db.workspace.dns_txt_records:
            self.db.workspace.dns_txt_records.append(value)
        return f"Published TXT record: {value}"

    @is_tool(ToolType.WRITE)
    def regenerate_api_key(self) -> str:
        """
        Regenerate your workspace API key. The old key stops working immediately.
        Only you can do this — support cannot rotate your key for you.

        Returns:
            Your new API key.
        """
        self.db.workspace.api_key_rotations += 1
        return api_key_for(
            self.db.workspace.account_id, self.db.workspace.api_key_rotations
        )

    @is_tool(ToolType.WRITE)
    def set_workspace_setting(self, setting: str, enabled: bool) -> dict:
        """
        Turn a workspace setting on or off.

        Args:
            setting: The setting name, e.g. 'double_opt_in' or 'campaign_sending_paused'.
            enabled: True to turn it on, False to turn it off.

        Returns:
            The updated settings.

        Raises:
            ValueError: If the setting does not exist in your workspace.
        """
        if setting not in self.db.workspace.settings:
            raise ValueError(
                f"No such setting '{setting}'. Available: "
                f"{sorted(self.db.workspace.settings)}"
            )
        self.db.workspace.settings[setting] = enabled
        return dict(self.db.workspace.settings)

    # ---------- assertion helpers for env_assertions (not tools) ----------

    def assert_api_key_rotations(self, expected: int) -> bool:
        """The API key has been rotated exactly `expected` times."""
        return self.db.workspace.api_key_rotations == expected

    def assert_dns_record_present(self, value: str) -> bool:
        """A given TXT record value is published."""
        return value.strip() in self.db.workspace.dns_txt_records

    def assert_setting(self, setting: str, expected: bool) -> bool:
        """A workspace setting has the expected value."""
        if setting not in self.db.workspace.settings:
            raise ValueError(f"No such setting '{setting}'")
        return self.db.workspace.settings[setting] is expected

    def add_email(self, email_id: str, sender: str, subject: str, body: str) -> Email:
        """Seed an email into the inbox (for initialization_actions)."""
        email = Email(email_id=email_id, sender=sender, subject=subject, body=body)
        self.db.inbox[email_id] = email
        return email

    def initialize_workspace(
        self,
        account_id: str,
        settings: dict | None = None,
        dns_txt_records: list | None = None,
        api_key_rotations: int = 0,
        emails: list | None = None,
    ) -> str:
        """Set up the customer's side of a task (for initialization_actions).

        tau2's other route to initial state, `initialization_data`, cannot be
        used by this domain: `Environment.set_state` reassigns
        `tools.db = user_tools.db` when `user_data` is present, which assumes
        the agent and the user share a single database object. This domain has
        two (that separation is what makes it dual-control), so all per-task
        setup goes through `initialization_actions` instead — the same route
        every telecom task uses.
        """
        self.db.workspace.account_id = account_id
        self.db.workspace.api_key_rotations = api_key_rotations
        self.db.workspace.dns_txt_records = list(dns_txt_records or [])
        if settings is not None:
            self.db.workspace.settings = dict(settings)
        self.db.inbox = {}
        for e in emails or []:
            self.add_email(
                email_id=e["email_id"],
                sender=e["sender"],
                subject=e["subject"],
                body=e["body"],
            )
        return f"Workspace initialized for {account_id}"
