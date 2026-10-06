"""Extract lease evidence and join it to MCL rows with transparent confidence."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from typing import Any

from .config import AUTHORIZED_ORGANIZATION_SIGNERS, IDENTIFIER_COLUMNS, USER_CONTEXT_COLUMNS
from .models import LeaseDocument, RenewalEvidence

DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y", "%m-%d-%Y")
DATE_PATTERN = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})\b")


def _clean_identifier(value: Any) -> str:
    """Normalize identifiers while preserving letters and digits for exact joins."""
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _clean_name(value: Any) -> str:
    """Normalize names for exact person checks without punctuation differences."""
    return " ".join(re.findall(r"[A-Z0-9]+", str(value or "").upper()))


def _name_in_values(name: str | None, values: list[Any]) -> bool:
    target = _clean_name(name)
    if not target:
        return False
    return any(target == _clean_name(value) or target in _clean_name(value) for value in values if value)


def _named_columns(row: dict[str, Any], purpose: str) -> list[Any]:
    """Pick source cells that can substantiate a named user or account holder."""
    selected = []
    for column, value in row.items():
        name = str(column).upper()
        if purpose == "user" and any(term in name for term in ("USER", "GOOD NAME", "INITIAL USER")) and not any(term in name for term in ("EMAIL", "STATUS")):
            selected.append(value)
        elif purpose == "holder" and any(term in name for term in ("ACCOUNT HOLDER", "PRIMARY ACCOUNT", "MAIN ACCOUNT", "OWNER")):
            selected.append(value)
    return selected


def _unit_count(row: dict[str, Any], product_terms: list[str]) -> tuple[int | None, str]:
    """Assante MCL rows each represent exactly one device, per user guidance."""
    return 1, "One device per MCL line item (per Assante workflow)"


def _find_value(text: str, labels: tuple[str, ...], pattern: str) -> str | None:
    """Return the first labeled identifier found in extracted document text."""
    label = "|".join(re.escape(item) for item in labels)
    match = re.search(rf"(?:{label})\s*[:#-]?\s*({pattern})", text, re.IGNORECASE)
    return match.group(1).strip(" .,#") if match else None


def _layout_label_value(text: str, label: str) -> str | None:
    """Read a value on the row below a PDF form label at the same column."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.search(rf"\b{re.escape(label)}\s*:", line, re.IGNORECASE)
        if match:
            column_start = match.start()
            for value_line in lines[index + 1:]:
                if not value_line.strip():
                    continue
                value = re.split(r"\s{2,}", value_line[column_start:].strip(), maxsplit=1)[0].strip()
                if value and not re.match(r"^(address|telephone no|email address|facsimile no|location of equipment)\s*:", value, re.IGNORECASE):
                    return value
    return None


def _fallback_evidence(document: LeaseDocument) -> RenewalEvidence:
    """Find common lease identifiers, term length, and a labeled expiry date."""
    text = document.text
    full_lease_id = re.search(r"\b\d{6,}-\d{3,}\b", f"{document.filename}\n{text}")
    lease_number = full_lease_id.group(0) if full_lease_id else _find_value(text, ("lease number", "lease #", "lease no", "lease addendum no"), r"[A-Z0-9][A-Z0-9/-]{3,}")
    sales_order = _find_value(text, ("sales order", "SO#", "order number", "order #"), r"[A-Z0-9][A-Z0-9/-]{3,}")
    company = _layout_label_value(text, "Lessee") or _find_value(text, ("customer", "leasee", "company"), r"[^\n\r|]{2,80}")
    account_holder = _find_value(text, ("main account holder", "primary account holder", "account holder", "account name", "bill to"), r"[^\n\r|]{2,80}")
    contact_name = _layout_label_value(text, "Contact") or _find_value(text, ("contact"), r"[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,3}")
    license_user = _find_value(text, ("license user", "licensed user", "user name", "user"), r"[^\n\r|]{2,80}")
    product = _find_value(text, ("product", "software", "license", "licence"), r"[^\n\r|]{2,100}")
    quantity = None
    product_line = re.search(r"(?m)^\s*(?P<product>[^\n]*\S)\s+\$[\d,]+\.\d{2}\s+(?P<quantity>\d+)\s+-?\$[\d,]+\.\d{2}\s+\$[\d,]+\.\d{2}\s*$", text)
    if product_line:
        product = product_line.group("product").strip()
        quantity = int(product_line.group("quantity"))
    date_match = re.search(r"(?:renewal|expiry|expiration|expires|end date|term end)[^\n\r]{0,80}?" + DATE_PATTERN.pattern, text, re.IGNORECASE)
    renewal_date = date_match.group(1) if date_match else None
    evidence_quote = text[max(0, date_match.start() - 70):date_match.end() + 40].replace("\n", " ") if date_match else None
    page = None
    if date_match:
        page_markers = list(re.finditer(r"\[page (\d+)\]", text[:date_match.start()], re.IGNORECASE))
        if page_markers:
            page = f"page {page_markers[-1].group(1)}"
    term_match = re.search(
        r"(?:initial\s+)?(?:lease\s+term|term(?:\s+of\s+(?:the\s+)?lease)?|lease\s+duration|term\s+length)"
        r"[^\n\r]{0,70}?(\d{1,3})\s*(months?|mos?\.?|years?|yrs?\.?)",
        text, re.IGNORECASE,
    )
    term_months = None
    if term_match:
        term_months = int(term_match.group(1)) * (12 if term_match.group(2).lower().startswith(("year", "yr")) else 1)
    else:
        layout_term = re.search(r"\(No\. of Months\)[^\n]*\n\s*(\d{1,3})\s+Monthly", text, re.IGNORECASE)
        if layout_term:
            term_months = int(layout_term.group(1))
    return RenewalEvidence(
        filename=document.filename, lease_number=lease_number, sales_order=sales_order,
        company=company.strip() if company else None,
        account_holder=account_holder.strip() if account_holder else None,
        contact_name=contact_name.strip() if contact_name else None,
        license_user=license_user.strip() if license_user else None,
        product=product.strip() if product else None, quantity=quantity,
        renewal_date=renewal_date, term_months=term_months, evidence_quote=evidence_quote, page=page,
    )


def _llm_evidence(document: LeaseDocument) -> RenewalEvidence | None:
    """Ask the configured model to extract labeled lease facts from one file."""
    if not os.getenv("OPENAI_API_KEY") or not document.text.strip():
        return None
    try:
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        extractor = model.with_structured_output(RenewalEvidence)
        result = extractor.invoke([
            ("system", "Extract only explicit facts from this lease text. renewal_date is an explicit software/service renewal or term-end date, not signature or shipment date. term_months is the initial lease term converted to months (for example, 3 years = 36); do not confuse payment frequency or equipment age with term length. Extract account_holder only when explicitly labeled as an account holder, contact_name from a Contact field, and license_user only when explicitly labeled. quantity is the number in the row for the named product, not a count of payments or months. Return an ISO date if clear. If a field is absent or ambiguous, leave it null. evidence_quote must quote the relevant date/term context verbatim. Do not infer identifiers."),
            ("human", f"Filename: {document.filename}\n\nLease text:\n{document.text[:45000]}"),
        ])
        result.filename = document.filename
        return result
    except Exception:
        return None


def extract_lease_evidence(raw_documents: list[dict[str, Any]]) -> list[RenewalEvidence]:
    """Extract dates and join keys from each parsed lease document."""
    evidence = []
    for raw in raw_documents:
        document = LeaseDocument(**raw)
        if document.error:
            evidence.append(RenewalEvidence(filename=document.filename, evidence_quote=f"Could not read file: {document.error}"))
            continue
        extracted = _llm_evidence(document)
        fallback = _fallback_evidence(document)
        if extracted is None:
            extracted = fallback
        else:
            # Keep model-extracted facts when present and fill only gaps with labeled text matches.
            for field in ("lease_number", "sales_order", "company", "account_holder", "contact_name", "license_user", "product", "quantity", "renewal_date", "term_months", "evidence_quote", "page"):
                if not getattr(extracted, field):
                    setattr(extracted, field, getattr(fallback, field))
        if extracted.renewal_date and not extracted.evidence_quote:
            extracted.evidence_quote = f"Extracted renewal/expiry date: {extracted.renewal_date}"
        evidence.append(extracted)
    return evidence


def _parse_date(value: Any) -> date | None:
    """Parse supported human-readable dates into calendar dates."""
    if value is None or not str(value).strip():
        return None
    value = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def _add_months(start: date, months: int) -> date:
    """Add calendar months while keeping the day valid for shorter months."""
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    # Constructing day 1 and adding the month length avoids an extra dependency.
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = (next_month - timedelta(days=1)).day
    return date(year, month, min(start.day, last_day))


def _lease_age_months(value: Any) -> float | None:
    """Interpret the MCL Lease Age (Year) decimal as elapsed years times 12."""
    try:
        age_years = float(str(value).replace(",", "").strip())
        return age_years * 12 if age_years >= 0 else None
    except (TypeError, ValueError):
        return None
def _license_columns(row: dict[str, Any]) -> list[str]:
    """Return likely software/license columns whose MCL cells contain a value."""
    license_columns = {
        "A4B", "BB (BE)", "BS", "BP", "E3", "TEAMS PREMIUM ('24)", "MS DEF. PLAN 1", "MS DEF. PLAN 2",
        "TEAMS VOICE", "ACCESS", "DOM. 120MINS", "DOM. 3000MINS", "INT. CALLING", "AUDIO CONF.",
        "PROJECT PLAN 5", "VISIO PLAN 2", "POWER BI", "(E3 - ACP HQ)", "(E3 - SLF HQ)",
        "TENANT TAKEOVER", "SHAREPOINT", "EXCHANGE", "PBX (LVL 1)", "PBX + IVR (LVL 3)",
    }
    found = []
    for column, value in row.items():
        name = str(column).upper()
        if name not in license_columns or not value:
            continue
        try:
            if float(str(value).replace(",", "")) <= 0:
                continue
        except (ValueError, TypeError):
            if str(value).strip().lower() not in {"yes", "y", "true", "x"}:
                continue
        found.append(str(column))
    return found


def _matching_evidence(row: dict[str, Any], evidence: list[RenewalEvidence], plan: dict[str, Any]) -> list[tuple[RenewalEvidence, str]]:
    """Match MCL rows against evidence using exact normalized identifiers."""
    row_ids = {_clean_identifier(row.get(column)) for column in IDENTIFIER_COLUMNS if row.get(column)}
    matches = []
    for item in evidence:
        evidence_ids = {_clean_identifier(value) for value in (item.lease_number, item.sales_order) if value}
        if row_ids & evidence_ids:
            matches.append((item, "Exact identifier"))
    if matches:
        return matches
    # Keep plausible documents visible for a human check when the PDF lacks a join key.
    requested_names = [name for name in (plan.get("target_user"), plan.get("account_holder")) if name]
    requested_products = [str(term).lower() for term in plan.get("product_terms", [])]
    for item in evidence:
        doc_names = [item.license_user, item.account_holder, item.contact_name, item.company]
        name_match = any(_name_in_values(name, doc_names) for name in requested_names)
        product_text = f"{item.product or ''} {item.filename}".lower()
        product_match = any(term in product_text for term in requested_products)
        if name_match or product_match:
            matches.append((item, "Candidate by document name or requested product; verify identifiers"))
    return matches


def build_renewal_results(
    rows: list[dict[str, Any]], evidence_items: list[RenewalEvidence], plan: dict[str, Any]
) -> list[dict[str, Any]]:
    """Filter MCL rows by query, match lease evidence, and flag uncertainty."""
    today = date.today()
    terms = [term.lower() for term in plan.get("product_terms", [])]
    days = plan.get("due_within_days")
    include_past = plan.get("include_past_due", False)
    output = []
    for row in rows:
        company = str(row.get("LEASEE/ COMPANY", ""))
        region = str(row.get("REGION", ""))
        if plan.get("company") and plan["company"].lower() not in company.lower():
            continue
        if plan.get("region") and plan["region"].lower() not in region.lower():
            continue
        if plan.get("target_user") and not _name_in_values(plan["target_user"], _named_columns(row, "user")):
            continue
        licenses = _license_columns(row)
        product_match = True
        product_check = "No specific software product requested"
        if terms:
            product_match = False
            product_values = []
            for column, value in row.items():
                column_name = str(column).strip().lower()
                if str(column).startswith("($)"):
                    continue
                if any(term == column_name for term in terms) and value and str(value).strip().lower() not in {"0", "0.0", "no", "false"}:
                    product_values.append(str(column))
            product_text = str(row.get("PRODUCT", "")).lower()
            if any(term in product_text for term in terms) or product_values:
                product_match = True
                product_check = "Requested product present: " + (", ".join(product_values) if product_values else str(row.get("PRODUCT", "")))
            else:
                product_check = "Requested product not found; matching MCL product cells are blank"
            if not product_match and not plan.get("target_user"):
                continue
        matched = _matching_evidence(row, evidence_items, plan)
        if not matched:
            unit_count, unit_basis = _unit_count(row, terms)
            status = "Review: requested product not found for this user" if not product_match else "Review: no matching lease document"
            result = _result_row(row, licenses, None, "No exact lease/SO match", status, None, unit_count=unit_count, unit_basis=unit_basis)
            result["REQUESTED PRODUCT CHECK"] = product_check
            output.append(result)
            continue
        # Return every linked lease document, leaving ambiguity visible rather than choosing one.
        for item, method in matched:
            expiry = _parse_date(item.renewal_date)
            date_basis = "Explicit renewal/expiry date in lease document" if expiry else ""
            lease_start = _parse_date(row.get("SHIPMENT DATE (LEASE START DATE)"))
            if expiry is None and item.term_months and lease_start:
                expiry = _add_months(lease_start, item.term_months)
                date_basis = f"Calculated from MCL lease start + {item.term_months}-month document term"
            age_months = _lease_age_months(row.get("LEASE AGE (YEAR)"))
            age_mismatch = False
            if lease_start and age_months is not None:
                elapsed_months = (today.year - lease_start.year) * 12 + (today.month - lease_start.month)
                age_mismatch = abs(elapsed_months - age_months) > 2
            mcl_holder_match = _name_in_values(plan.get("account_holder"), _named_columns(row, "holder")) if plan.get("account_holder") else None
            pdf_holder_match = _name_in_values(plan.get("account_holder"), [item.contact_name, item.account_holder]) if plan.get("account_holder") else None
            pdf_user_match = _name_in_values(plan.get("target_user"), [item.license_user]) if plan.get("target_user") else None
            association = _association_text(method, mcl_holder_match, pdf_holder_match, pdf_user_match, row, item, plan)
            unit_count, unit_basis = _unit_count(row, terms)
            if expiry is None:
                result = _result_row(row, licenses, item, method, "Review: no explicit expiry date or usable term/start date", None, date_basis, unit_count, unit_basis, association)
                result["REQUESTED PRODUCT CHECK"] = product_check
                output.append(result)
                continue
            remaining = (expiry - today).days
            if days is not None and (remaining > days or (remaining < 0 and not include_past)):
                continue
            if days is None and remaining < 0 and not include_past:
                continue
            if age_mismatch:
                output.append(_result_row(row, licenses, item, method, "Review: Lease Age does not align with lease start date", expiry, date_basis, unit_count, unit_basis, association))
                continue
            status = (
                "Review: projected lease term end; confirm software renewal date"
                if date_basis.startswith("Calculated")
                else "Matched; confirm source terms"
            )
            if method != "Exact identifier" or association.startswith("Mismatch") or not product_match:
                status = "Review: document association needs confirmation"
            result = _result_row(row, licenses, item, method, status, expiry, date_basis, unit_count, unit_basis, association)
            result["REQUESTED PRODUCT CHECK"] = product_check
            output.append(result)
    return output


def _association_text(method: str, mcl_holder: bool | None, pdf_holder: bool | None, pdf_user: bool | None, row: dict[str, Any], item: RenewalEvidence, plan: dict[str, Any]) -> str:
    """Summarize exactly which requested names were corroborated by source cells."""
    checks = []
    company_matches = _clean_name(row.get("LEASEE/ COMPANY")) == _clean_name(item.company)
    if company_matches:
        checks.append(f"organization matches: {item.company}")
    if plan.get("account_holder"):
        company_key = _clean_name(item.company)
        authorized_signers = next(
            (signers for organization, signers in AUTHORIZED_ORGANIZATION_SIGNERS.items() if _clean_name(organization) == company_key),
            set(),
        )
        expected_signer = _clean_name(plan["account_holder"])
        if pdf_holder and expected_signer in {_clean_name(name) for name in authorized_signers}:
            checks.append(f"authorized signatory matches: {plan['account_holder']}")
        elif pdf_holder:
            checks.append(f"PDF contact matches {plan['account_holder']}; signer role is not configured")
        else:
            checks.append(f"PDF does not identify {plan['account_holder']} as contact or account holder")
    if plan.get("target_user"):
        checks.append(f"PDF does not name {plan['target_user']} as licensed user" if not pdf_user else f"PDF licensed user matches {plan['target_user']}")
    specific_products = [str(term).lower() for term in plan.get("product_terms", []) if str(term).lower() not in {"software", "license", "licence"}]
    if specific_products:
        for product in specific_products:
            mcl_product = product in str(row.get("PRODUCT", "")).lower() or any(
                product == str(column).strip().lower()
                and str(value).strip().lower() not in {"", "0", "0.0", "no", "false"}
                for column, value in row.items() if not str(column).startswith("($)")
            )
            checks.append(f"MCL {product} assignment {'present' if mcl_product else 'blank'}")
            if product in str(item.product or "").lower():
                checks.append(f"PDF product identifies {product}")
            else:
                checks.append(f"PDF product is {item.product or 'not stated'}, which does not identify {product}")
    if method != "Exact identifier":
        checks.append("separate lease number; this is allowed for a separate software-license lease")
    if not checks:
        return "Not checked"
    return "Association review: " + "; ".join(checks)


def _result_row(row: dict[str, Any], licenses: list[str], item: RenewalEvidence | None, method: str, status: str, expiry: date | None, date_basis: str = "", unit_count: int | None = None, unit_basis: str = "", association: str = "") -> dict[str, Any]:
    """Shape a compact row with MCL context and document-level provenance."""
    result = {column: row.get(column, "") for column in USER_CONTEXT_COLUMNS if column in row}
    result["SOFTWARE LICENSES"] = ", ".join(licenses)
    result["UNIT COUNT"] = unit_count if unit_count is not None else ""
    result["UNIT COUNT BASIS"] = unit_basis or "One matching MCL row; no explicit unit quantity column found"
    result["MCL TOTAL COUNT"] = row.get("TOTAL COUNT", "")
    result["MCL INITIAL USER"] = row.get("INITIAL USER NAME", "")
    result["MCL ROW ID"] = row.get("IDX", "")
    result["RENEWAL / TERM END DATE"] = expiry.isoformat() if expiry else ((item.renewal_date or "") if item else "")
    result["LEASE TERM (MONTHS)"] = item.term_months if item else ""
    result["DATE BASIS"] = date_basis
    result["LEASE MATCH"] = method
    result["REVIEW STATUS"] = status
    result["SOURCE DOCUMENT"] = item.filename if item else ""
    result["SOURCE PAGE"] = item.page if item and item.page else ""
    result["SOURCE EVIDENCE"] = item.evidence_quote if item and item.evidence_quote else ""
    result["DOCUMENT ASSOCIATION CHECK"] = association
    holder_values = [value for value in _named_columns(row, "holder") if value]
    result["MCL ACCOUNT HOLDER"] = "; ".join(dict.fromkeys(str(value).strip() for value in holder_values))
    result["PDF ACCOUNT HOLDER"] = item.account_holder or item.company or "" if item else ""
    result["PDF LICENSE USER"] = item.license_user or "" if item else ""
    result["PDF PRODUCT"] = item.product or "" if item else ""
    result["PDF QUANTITY"] = item.quantity if item and item.quantity is not None else ""
    result["PDF CONTACT"] = item.contact_name or "" if item else ""
    result["PDF LEASE ID"] = item.lease_number or "" if item else ""
    return result


def make_summary(rows_checked: int, results: list[dict[str, Any]]) -> dict[str, int]:
    """Count results and records requiring a human review."""
    return {
        "mcl_rows_checked": rows_checked,
        "matched_rows": len(results),
        "review_rows": sum(1 for row in results if str(row.get("REVIEW STATUS", "")).startswith("Review")),
    }
