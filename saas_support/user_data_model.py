"""User-side database: what the *customer* can see and do in their own workspace.

This is the half of the environment the support agent cannot reach. The agent
can read Loopline's console; only the customer can read their inbox, rotate
their API key, edit their DNS, or flip a setting inside their workspace. Tasks
in this domain are built so that neither side can finish alone.
"""

from typing import Dict, List

from pydantic import Field

from tau2.environment.db import DB
from tau2.utils.pydantic_utils import BaseModelNoExtra


class Email(BaseModelNoExtra):
    """A message sitting in the customer's inbox."""

    email_id: str = Field(description="Unique identifier for the email")
    sender: str = Field(description="Who sent the email")
    subject: str = Field(description="Subject line")
    body: str = Field(description="Body text")


class Workspace(BaseModelNoExtra):
    """The customer's own Loopline workspace."""

    account_id: str = Field(description="Account this workspace belongs to")
    api_key_rotations: int = Field(
        default=0, description="How many times the API key has been regenerated"
    )
    dns_txt_records: List[str] = Field(
        default_factory=list,
        description="TXT records currently published on the sending domain",
    )
    settings: Dict[str, bool] = Field(
        default_factory=dict, description="Workspace toggles the customer controls"
    )


class SaasUserDB(DB):
    """Everything on the customer's side of the conversation."""

    workspace: Workspace = Field(description="The customer's workspace")
    inbox: Dict[str, Email] = Field(
        default_factory=dict, description="Customer inbox indexed by email ID"
    )

    def get_statistics(self) -> dict:
        return {
            "num_emails": len(self.inbox),
            "num_dns_records": len(self.workspace.dns_txt_records),
            "num_settings": len(self.workspace.settings),
        }
