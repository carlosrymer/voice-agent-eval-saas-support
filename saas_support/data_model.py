"""Agent-side database for the Loopline B2B SaaS support domain.

Design note: nothing the *agent* free-types is ever persisted here. Every
write tool takes enumerated codes rather than prose, because tau2 scores the
DB reward by comparing a hash of this model against a hash produced by
replaying the task's reference trajectory. A free-text `reason` field would
mean the agent could solve a task perfectly and still mismatch the gold hash
purely because it phrased a note differently.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from tau2.environment.db import DB

AccountStatus = Literal["active", "past_due", "suspended"]
InvoiceStatus = Literal["paid", "open", "refunded"]
TicketStatus = Literal["open", "closed"]
TicketPriority = Literal["low", "normal", "high", "urgent"]

CreditReasonCode = Literal[
    "service_outage",
    "billing_error",
    "goodwill",
    "overcharge",
]

EscalationCategory = Literal[
    "refund_over_threshold",
    "plan_exception",
    "security_incident",
    "other",
]


class Plan(BaseModel):
    plan_id: str = Field(description="Unique identifier for the plan")
    name: str = Field(description="Display name of the plan")
    price_per_seat_cents: int = Field(description="Monthly price per seat, in cents")
    max_daily_sends: int = Field(
        description="Hard ceiling on the daily campaign sending limit for this plan"
    )
    features: List[str] = Field(
        default_factory=list, description="Feature entitlements included in this plan"
    )


class ScheduledSeatChange(BaseModel):
    new_seats: int = Field(description="Seat count that takes effect next cycle")
    effective_date: str = Field(description="ISO date the change takes effect")


class Account(BaseModel):
    account_id: str = Field(description="Unique identifier for the account")
    company_name: str = Field(description="Customer company name")
    plan_id: str = Field(description="The plan this account is subscribed to")
    seats_purchased: int = Field(description="Seats currently billed")
    seats_used: int = Field(description="Seats currently assigned to users")
    billing_cycle_start: str = Field(description="ISO date the current cycle started")
    billing_cycle_end: str = Field(description="ISO date the current cycle ends")
    status: AccountStatus = Field(description="Account standing")
    daily_send_limit: int = Field(description="Current campaign sends allowed per day")
    sending_domain: str = Field(description="Domain campaigns are sent from")
    domain_verified: bool = Field(
        default=False,
        description="Whether the sending domain's DNS TXT record has been confirmed",
    )
    owner_name: str = Field(description="Name of the account owner")
    owner_email: str = Field(description="Email of the account owner")
    authorized_contacts: List[str] = Field(
        default_factory=list,
        description="Emails permitted to make changes to this account",
    )
    security_code: str = Field(
        description="Static identity-verification code emailed to the account owner"
    )
    credit_balance_cents: int = Field(
        default=0, description="Account credit balance, in cents"
    )
    scheduled_seat_change: Optional[ScheduledSeatChange] = Field(
        default=None, description="Pending seat change for the next billing cycle"
    )


class Invoice(BaseModel):
    invoice_id: str = Field(description="Unique identifier for the invoice")
    account_id: str = Field(description="Account the invoice belongs to")
    amount_cents: int = Field(description="Invoice total, in cents")
    period: str = Field(description="Billing period covered, e.g. 2026-02")
    status: InvoiceStatus = Field(description="Invoice status")


class Credit(BaseModel):
    credit_id: str = Field(description="Unique identifier for the credit")
    account_id: str = Field(description="Account the credit was applied to")
    amount_cents: int = Field(description="Credit amount, in cents")
    reason_code: CreditReasonCode = Field(description="Why the credit was issued")


class Escalation(BaseModel):
    escalation_id: str = Field(description="Unique identifier for the escalation")
    account_id: str = Field(description="Account the escalation concerns")
    category: EscalationCategory = Field(description="Escalation category")
    requested_amount_cents: int = Field(
        default=0,
        description="Amount requested, in cents. 0 when not a monetary request.",
    )


class Ticket(BaseModel):
    ticket_id: str = Field(description="Unique identifier for the ticket")
    account_id: str = Field(description="Account the ticket belongs to")
    subject: str = Field(description="Ticket subject")
    status: TicketStatus = Field(description="Ticket status")
    priority: TicketPriority = Field(description="Ticket priority")


class SaasDB(DB):
    """Loopline's internal support console database."""

    plans: Dict[str, Plan] = Field(description="Plans indexed by plan ID")
    accounts: Dict[str, Account] = Field(description="Accounts indexed by account ID")
    invoices: Dict[str, Invoice] = Field(description="Invoices indexed by invoice ID")
    credits: Dict[str, Credit] = Field(
        default_factory=dict, description="Issued credits indexed by credit ID"
    )
    escalations: Dict[str, Escalation] = Field(
        default_factory=dict, description="Escalations indexed by escalation ID"
    )
    tickets: Dict[str, Ticket] = Field(
        default_factory=dict, description="Tickets indexed by ticket ID"
    )

    def get_statistics(self) -> dict:
        return {
            "num_plans": len(self.plans),
            "num_accounts": len(self.accounts),
            "num_invoices": len(self.invoices),
            "num_credits": len(self.credits),
            "num_escalations": len(self.escalations),
            "num_tickets": len(self.tickets),
        }
