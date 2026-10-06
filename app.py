"""Streamlit front end for the MCL software renewal assistant."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from mcl_agent.config import DEFAULT_LEASE_FOLDER
from mcl_agent.exporter import create_results_workbook
from mcl_agent.graph import build_agent_graph
from mcl_agent.ingestion import load_mcl_workbook
from mcl_agent.lease_reader import load_lease_documents
from mcl_agent.observability import analysis_run, node_span


def _save_uploaded_files(uploaded_files: list, destination: Path) -> list[Path]:
    """Save browser uploads to a temporary directory and return their paths."""
    destination.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for uploaded in uploaded_files:
        target = destination / Path(uploaded.name).name
        target.write_bytes(uploaded.getvalue())
        saved_paths.append(target)
    return saved_paths


def main() -> None:
    """Render the configuration form and run the renewal analysis on request."""
    st.set_page_config(page_title="MCL Renewal Assistant", page_icon="📄", layout="wide")
    st.title("MCL software renewal assistant")
    st.caption("Ask a question, check lease evidence, and download a traceable Excel report.")

    with st.sidebar:
        st.header("Your source files")
        mcl_upload = st.file_uploader("MCL workbook (.xlsx)", type=["xlsx"])
        default_folder = str(DEFAULT_LEASE_FOLDER)
        lease_folder = st.text_input("Lease Documents folder", value=default_folder)
        pdf_uploads = st.file_uploader(
            "Upload lease PDFs",
            type=["pdf"], accept_multiple_files=True,
            help="Text-based PDF files are read page by page and included in the renewal analysis.",
        )
        other_lease_uploads = st.file_uploader(
            "Optional lease documents (.docx, .txt)",
            type=["docx", "txt"], accept_multiple_files=True,
        )
        st.caption("The folder path is read by this local app process. Uploaded files are read for this analysis.")

    query = st.text_area(
        "What do you want to find?",
        placeholder="Example: Which users at each company have Microsoft software renewals in the next 90 days?",
        height=90,
    )
    run = st.button("Find renewals", type="primary", disabled=not bool(query.strip()))

    if not run:
        st.info("Choose the MCL workbook, confirm the lease folder, and enter a question to begin.")
        st.markdown("**The report shows:** matching MCL users and license columns, renewal dates, lease-document evidence, and records that need manual review.")
        return

    if mcl_upload is None:
        st.error("Upload the MCL workbook before running a query.")
        return

    with tempfile.TemporaryDirectory(prefix="mcl-renewal-") as temp_name:
        temp_dir = Path(temp_name)
        mcl_path = temp_dir / Path(mcl_upload.name).name
        mcl_path.write_bytes(mcl_upload.getvalue())
        uploaded_pdf_paths = _save_uploaded_files(pdf_uploads or [], temp_dir / "uploaded-pdfs")
        uploaded_other_paths = _save_uploaded_files(other_lease_uploads or [], temp_dir / "uploaded-other")

        with st.spinner("Reading MCL records and lease documents…"):
            try:
                mcl_frame = load_mcl_workbook(mcl_path)
                lease_docs = load_lease_documents(
                    Path(lease_folder).expanduser(), uploaded_pdf_paths + uploaded_other_paths
                )
                if not lease_docs:
                    st.warning("No lease documents were found. The MCL query will still run, but renewal dates and lease matches will be marked for review.")
                pdf_docs = [doc for doc in lease_docs if Path(doc.filename).suffix.lower() == ".pdf"]
                if pdf_docs:
                    st.subheader("PDF reading status")
                    for doc in pdf_docs:
                        if doc.error:
                            st.error(f"{doc.filename}: could not read this PDF ({doc.error})")
                        elif not doc.text.strip():
                            st.warning(f"{doc.filename}: no selectable text was found. This may be a scanned PDF and needs OCR.")
                        else:
                            st.success(f"{doc.filename}: read {len(doc.page_labels)} page(s), {len(doc.text):,} characters extracted.")
                        if doc.text.strip():
                            with st.expander(f"View extracted text: {doc.filename}"):
                                st.text(doc.text)
                graph = build_agent_graph()
                with analysis_run(
                    query_chars=len(query),
                    mcl_row_count=len(mcl_frame),
                    document_count=len(lease_docs),
                ) as tracker:
                    graph_input = {
                        "query": query,
                        "mcl_rows": mcl_frame.to_dict(orient="records"),
                        "lease_documents": [doc.model_dump() for doc in lease_docs],
                    }
                    with node_span("renewal_analysis", graph_input) as span:
                        result = graph.invoke(graph_input)
                        output_bytes = create_results_workbook(result["results"], result["summary"])
                        tracker.log_result(
                            mcl_rows_checked=result["summary"]["mcl_rows_checked"],
                            matched_rows=result["summary"]["matched_rows"],
                            review_rows=result["summary"]["review_rows"],
                            lease_documents_read=sum(not doc.error for doc in lease_docs),
                            lease_documents_failed=sum(bool(doc.error) for doc in lease_docs),
                        )
                        span.set_outputs({
                            "answer": result["answer"],
                            "results": result["results"],
                            "summary": result["summary"],
                            "steps": result["steps"],
                            "report": {
                                "filename": "mcl_software_renewals.xlsx",
                                "bytes": len(output_bytes),
                            },
                        })
            except Exception as exc:  # Show actionable input and configuration errors in the UI.
                st.error(f"Analysis could not be completed: {exc}")
                st.exception(exc)
                return

    summary = result["summary"]
    if tracker.enabled and not tracker.span_failures:
        st.caption(f"Analysis and traces recorded in MLflow (run `{tracker.run_id}`).")
        st.link_button("Open this experiment’s traces", tracker.traces_url)
    elif tracker.enabled:
        st.warning(f"MLflow run `{tracker.run_id}` was recorded, but trace spans could not all be saved.")
        if tracker.failure_reason:
            st.caption(tracker.failure_reason)
        st.link_button("Open this experiment’s traces", tracker.traces_url)
    else:
        reason = tracker.failure_reason or "MLflow was unavailable."
        st.warning(f"This analysis completed without MLflow telemetry. {reason}")
        st.link_button("Open MLflow", tracker.tracking_uri)
    c1, c2, c3 = st.columns(3)
    c1.metric("MCL rows checked", summary["mcl_rows_checked"])
    c2.metric("Matching users / records", summary["matched_rows"])
    c3.metric("Needs review", summary["review_rows"])
    st.write(result["answer"])
    st.dataframe(result["results"], use_container_width=True, hide_index=True)
    st.download_button(
        "Download Excel report",
        data=output_bytes,
        file_name="mcl_software_renewals.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
    with st.expander("How the agent reached this result"):
        st.write(" → ".join(result["steps"]))
        st.write("Lease renewal dates are taken from document text and joined to MCL using lease number, sales order, serial number, or company. Unmatched or ambiguous evidence is marked for review.")


if __name__ == "__main__":
    main()
