"""Turn natural-language questions into constrained, auditable query filters."""

from __future__ import annotations

import os
import re
from datetime import date

from .models import QueryPlan


def _rule_based_plan(query: str) -> QueryPlan:
    """Parse common renewal time-window phrases when no LLM is configured."""
    lowered = query.lower()
    within = re.search(r"(?:next|within|coming)\s+(\d+)\s+days?", lowered)
    if within:
        days = int(within.group(1))
    elif "this month" in lowered:
        days = (date.today().replace(day=28) - date.today()).days + 4
    elif "next 3 months" in lowered or "next quarter" in lowered:
        days = 90
    elif "next 6 months" in lowered:
        days = 180
    else:
        days = None
    past_due = any(term in lowered for term in ("past due", "overdue", "already expired"))
    target_match = re.search(
        r"how many (?:units|licenses|licences)\s+(?:(?:does|do)\s+)?(?:the\s+)?(?:user\s+)?(.+?)(?:\s+(?:have|own|hold)\b)",
        query, re.IGNORECASE,
    )
    if not target_match:
        target_match = re.search(r"\bthe\s+user\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,2})", query)
    if not target_match:
        target_match = re.search(r"\bfor\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,2})", query)
    holder_match = re.search(
        r"\b(?:main|primary)\s+account\s+holder\s+(?:is|:)?\s*([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,3})",
        query, re.IGNORECASE,
    )
    product_terms = [term for term in ("software", "license", "microsoft", "teams", "visio", "power bi", "exchange", "sharepoint") if term in lowered]
    if re.search(r"\bbp\b", lowered):
        product_terms.append("BP")
    return QueryPlan(
        query_summary=query.strip(), due_within_days=days,
        include_past_due=past_due,
        product_terms=product_terms,
        target_user=target_match.group(1).strip(" ,.?!") if target_match else None,
        account_holder=holder_match.group(1).strip(" ,.?!") if holder_match else None,
    )


def parse_query(query: str) -> QueryPlan:
    """Use an optional LLM for structured filters, falling back safely to rules."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _rule_based_plan(query)
    try:
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=1,
            api_key=api_key,
        )
        parser = model.with_structured_output(QueryPlan)
        parsed = parser.invoke([
            ("system", "Extract search filters and named entities from the user question. Only set a time window when stated or clearly implied. Do not invent company or region values. Use product_terms for explicit product/software concepts, target_user for the person whose records are requested, and account_holder for an explicitly named main/primary account holder. Preserve names as written. If a value is ambiguous, leave it empty."),
            ("human", query),
        ])
        rules = _rule_based_plan(query)
        if not parsed.target_user:
            parsed.target_user = rules.target_user
        if not parsed.account_holder:
            parsed.account_holder = rules.account_holder
        if not parsed.product_terms:
            parsed.product_terms = rules.product_terms
        return parsed
    except Exception:
        # A provider/network/structured-output issue should not prevent a local query.
        return _rule_based_plan(query)
