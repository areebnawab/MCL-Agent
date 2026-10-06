"""Typed data structures shared across the LangGraph nodes."""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    """Structured filters inferred from a user's plain-language question."""
    query_summary: str = Field(description="Short plain-language description of the requested records")
    due_within_days: Optional[int] = Field(default=None, description="Include expiries due within this many days")
    region: Optional[str] = None
    company: Optional[str] = None
    product_terms: list[str] = Field(default_factory=list)
    include_past_due: bool = Field(default=False)
    target_user: Optional[str] = Field(default=None, description="Specific MCL user named in the question")
    account_holder: Optional[str] = Field(default=None, description="Main or primary account holder named in the question")


class LeaseDocument(BaseModel):
    """Text extracted from a source file, with its original filename for citations."""
    filename: str
    text: str
    page_labels: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class RenewalEvidence(BaseModel):
    """Renewal details found in one document, kept with source evidence."""
    filename: str = ""
    lease_number: Optional[str] = None
    sales_order: Optional[str] = None
    company: Optional[str] = None
    account_holder: Optional[str] = None
    contact_name: Optional[str] = None
    license_user: Optional[str] = None
    product: Optional[str] = None
    quantity: Optional[int] = None
    renewal_date: Optional[str] = None
    term_months: Optional[int] = None
    evidence_quote: Optional[str] = None
    page: Optional[str] = None


class AgentState(TypedDict, total=False):
    """State passed through the query, extraction, matching, and response nodes."""
    query: str
    mcl_rows: list[dict[str, Any]]
    lease_documents: list[dict[str, Any]]
    query_plan: dict[str, Any]
    lease_evidence: list[dict[str, Any]]
    results: list[dict[str, Any]]
    summary: dict[str, int]
    answer: str
    steps: list[str]
