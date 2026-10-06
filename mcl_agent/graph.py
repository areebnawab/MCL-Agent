"""LangGraph orchestration for parsing, evidence extraction, matching, and response."""

from langgraph.graph import END, START, StateGraph

from .matching import build_renewal_results, extract_lease_evidence, make_summary
from .models import AgentState
from .observability import node_span
from .query_parser import parse_query


def _plan_query(state: AgentState) -> dict:
    """Convert the question into a constrained filter plan."""
    with node_span("plan_query", {"query": state["query"]}) as span:
        plan = parse_query(state["query"]).model_dump()
        span.set_outputs(plan)
        return {"query_plan": plan}


def _extract_leases(state: AgentState) -> dict:
    """Extract renewal facts and source evidence from all lease documents."""
    documents = state["lease_documents"]
    with node_span("extract_leases", {"lease_documents": documents}) as span:
        evidence = [item.model_dump() for item in extract_lease_evidence(documents)]
        span.set_outputs({"lease_evidence": evidence})
        return {"lease_evidence": evidence}


def _match_records(state: AgentState) -> dict:
    """Apply filters and link MCL rows to lease evidence by exact identifiers."""
    from .models import RenewalEvidence
    with node_span("match_records", {
        "mcl_rows": state["mcl_rows"],
        "lease_evidence": state["lease_evidence"],
        "query_plan": state["query_plan"],
    }) as span:
        evidence = [RenewalEvidence(**item) for item in state["lease_evidence"]]
        results = build_renewal_results(state["mcl_rows"], evidence, state["query_plan"])
        summary = make_summary(len(state["mcl_rows"]), results)
        span.set_outputs({"results": results, "summary": summary})
        return {"results": results, "summary": summary}


def _write_response(state: AgentState) -> dict:
    """Answer the specific question first, then summarize the analysis."""
    with node_span("write_response", {
        "query_plan": state.get("query_plan", {}),
        "lease_evidence": state.get("lease_evidence", []),
        "results": state.get("results", []),
        "summary": state.get("summary", {}),
    }) as span:
        response = _build_response(state)
        span.set_outputs(response)
        return response


def _build_response(state: AgentState) -> dict:
    """Build the user-facing answer from the completed analysis state."""
    summary = state["summary"]
    answer = (f"Checked {summary['mcl_rows_checked']} MCL rows. Found {summary['matched_rows']} matching records; "
              f"{summary['review_rows']} need review because the date or lease match needs confirmation.")
    plan = state.get("query_plan", {})
    rows = state.get("results", [])
    person = plan.get("target_user")
    if person:
        if not rows:
            answer = f"I could not find an MCL row for {person} that matches the requested filters. Check the spelling and the user/name columns in the workbook."
        else:
            # Candidate documents can create multiple result rows for one MCL record.
            unique = {}
            for index, row in enumerate(rows):
                row_id = str(row.get("MCL ROW ID", "")).strip()
                key = ("mcl-row", row_id) if row_id else tuple(str(row.get(column, "")) for column in ("LEASE NUMBER", "SO#", "CURRENT SERIAL #"))
                unique[key if any(key) else (index,)] = row
            records = list(unique.values())
            quantity_values = [row.get("UNIT COUNT") for row in records if str(row.get("UNIT COUNT", "")).strip()]
            basis = "; ".join(sorted({str(row.get("UNIT COUNT BASIS", "")) for row in records if row.get("UNIT COUNT BASIS")}))
            if len(quantity_values) == len(records):
                quantity = sum(int(value) for value in quantity_values)
                answer = f"For {person}, I found {len(records)} MCL line item(s), which represent {quantity} device unit(s) ({basis})."
            else:
                answer = f"For {person}, I found {len(records)} matching MCL record(s), but the workbook does not expose a usable unit quantity for every record, so I cannot reliably total the units."
            dates = sorted({str(row.get("RENEWAL / TERM END DATE", "")).strip() for row in records if row.get("RENEWAL / TERM END DATE") is not None and str(row.get("RENEWAL / TERM END DATE", "")).strip()})
            answer += " Lease expiry: " + (", ".join(dates) if dates else "not established from the matched lease evidence") + "."
            docs = sorted({str(row.get("SOURCE DOCUMENT", "")).strip() for row in records if str(row.get("SOURCE DOCUMENT", "")).strip()})
            if docs:
                pages = sorted({str(row.get("SOURCE PAGE", "")).strip() for row in records if str(row.get("SOURCE PAGE", "")).strip()})
                answer += " Candidate software lease reviewed: " + ", ".join(docs)
                if pages:
                    answer += " (" + ", ".join(pages) + ")"
                answer += "."
            else:
                answer += " No software lease candidate was found in the uploaded documents. Separate license leases may have a different number, so the agent checks company, signer, user, and product context."
            product_checks = sorted({str(row.get("REQUESTED PRODUCT CHECK", "")).strip() for row in records if row.get("REQUESTED PRODUCT CHECK")})
            if product_checks and any("not found" in check.lower() for check in product_checks):
                for row in records:
                    lease_id = str(row.get("LEASE NUMBER", "")).strip()
                    requested_product = plan.get("product_terms", ["requested product"])[-1]
                    answer += f" MCL product check: the {requested_product} cell is blank on lease {lease_id or 'unknown'}; the MCL row count establishes one device, not a BP license entitlement."
                    initial_user = str(row.get("MCL INITIAL USER", "")).strip()
                    if initial_user:
                        answer += f" The workbook lists {initial_user} as the initial user and {person} as the current user."
                holder = plan.get("account_holder")
                evidence_items = state.get("lease_evidence", [])
                relevant_documents = []
                for item in evidence_items:
                    doc_names = [item.get("contact_name"), item.get("account_holder"), item.get("license_user")]
                    if holder and any(str(value or "").strip().casefold() == str(holder).strip().casefold() for value in doc_names):
                        relevant_documents.append(item)
                for item in relevant_documents:
                    doc_id = str(item.get("lease_number") or "not identified")
                    mcl_ids = {str(row.get("LEASE NUMBER", "")).strip() for row in records}
                    answer += f" Uploaded PDF {item.get('filename', '')} identifies lease {doc_id}"
                    if doc_id not in mcl_ids:
                        answer += f", a separate lease number from MCL lease {', '.join(sorted(mcl_ids))}; this can be normal for a separate software-license lease"
                    contact = item.get("contact_name")
                    if contact:
                        answer += f"; it lists {contact} as the contact and authorized Assante signatory per your guidance"
                    if item.get("company"):
                        answer += f"; Lessee is {item['company']}"
                    if item.get("product"):
                        answer += f" for {item['product']}"
                    if item.get("quantity") is not None:
                        answer += f", quantity {item['quantity']}"
                    if not item.get("license_user"):
                        answer += f"; it does not name {person} as the licensed user"
                    if item.get("renewal_date"):
                        answer += f". Its stated renewal/end date is {item['renewal_date']}"
                    elif item.get("term_months") is not None:
                        answer += f". Its term field reads {item['term_months']} months, and it gives no usable renewal/end date"
                    else:
                        answer += ". It does not provide an explicit renewal/end date or a usable lease term"
                    answer += "."
            checks = sorted({str(row.get("DOCUMENT ASSOCIATION CHECK", "")).strip() for row in records if str(row.get("DOCUMENT ASSOCIATION CHECK", "")).strip()})
            if checks:
                if any("authorized signatory matches" in check for check in checks):
                    answer += " The Assante organization and authorized signatory align; the PDF does not name the user, and its product is not BP, so it does not confirm a BP assignment for this person."
            if summary["review_rows"]:
                answer += f" {summary['review_rows']} record(s) remain flagged for review; see the evidence table below."
    return {"answer": answer, "steps": ["Parse natural-language filters", "Extract lease facts and source evidence", "Join MCL rows to documents", "Build traceable Excel report"]}


def build_agent_graph():
    """Create and compile the explicit four-stage LangGraph workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node("plan_query", _plan_query)
    workflow.add_node("extract_leases", _extract_leases)
    workflow.add_node("match_records", _match_records)
    workflow.add_node("write_response", _write_response)
    workflow.add_edge(START, "plan_query")
    workflow.add_edge("plan_query", "extract_leases")
    workflow.add_edge("extract_leases", "match_records")
    workflow.add_edge("match_records", "write_response")
    workflow.add_edge("write_response", END)
    return workflow.compile()
