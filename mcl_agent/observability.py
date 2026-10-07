"""MLflow run tracking and full input/output tracing for renewal analyses."""

from __future__ import annotations

import os
import sys
import time
import math
import logging
from datetime import date, datetime
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Any, Iterator

import mlflow
from mlflow.entities import SpanType


DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "MCL Renewal Agent"
logger = logging.getLogger(__name__)
_ACTIVE_TRACKER: ContextVar["AnalysisTracker | None"] = ContextVar("mcl_mlflow_tracker", default=None)


def configure_mlflow() -> str:
    """Point MLflow at the configured tracking server and experiment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT))
    # ChatOpenAI is invoked through LangChain; its tracing integration records
    # nested LLM spans and provider-reported token usage.
    try:
        mlflow.langchain.autolog(log_traces=True, silent=True)
    except Exception:
        # Keep telemetry best-effort if the integration is unavailable or the
        # installed LangChain version is unsupported.
        pass
    return experiment.experiment_id


def _tracking_ui_url(tracking_uri: str) -> str:
    """Return a browser URL for the tracking server, not its MLflow URI."""
    if tracking_uri == "databricks":
        # `databricks` is an MLflow protocol URI, not a browser address.
        return os.getenv("DATABRICKS_HOST", "").rstrip("/")
    return tracking_uri.rstrip("/")


def _experiment_traces_url(tracking_uri: str, experiment_id: str) -> str:
    """Build the experiment's trace page URL for the configured tracking server."""
    if tracking_uri == "databricks":
        host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
        return f"{host}/ml/experiments/{experiment_id}/traces" if host else ""
    return f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}/traces?workflowType=genai"


@contextmanager
def analysis_run(
    *, query_chars: int, mcl_row_count: int, document_count: int
) -> Iterator["AnalysisTracker"]:
    """Create one run per analysis, falling back to local operation if MLflow is down.

    Run metadata stays aggregate-only. Full request/response payloads are attached to
    explicit spans by the caller for debugging and evaluation.
    """
    tracker = AnalysisTracker()
    tracker.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    tracker.tracking_url = _tracking_ui_url(tracker.tracking_uri)
    try:
        experiment_id = configure_mlflow()
        tracker.traces_url = _experiment_traces_url(tracker.tracking_uri, experiment_id)
        run_context = mlflow.start_run(run_name="renewal-analysis")
        run = run_context.__enter__()
        tracker.run_id = run.info.run_id
    except Exception as exc:
        # Keep the browser message credential-safe while preserving the
        # Databricks REST response in Railway's server logs for diagnosis.
        logger.exception("Could not initialize the MLflow experiment or open a run")
        tracker.failure_reason = f"{type(exc).__name__} while opening the MLflow run at {tracker.tracking_uri}"
        yield tracker
        return

    tracker.enabled = True
    tracker.started_at = time.perf_counter()
    try:
        try:
            mlflow.set_tags({"app": "mcl-renewal-agent", "observability": "full-agent-io"})
            mlflow.log_params({
                "query_chars": query_chars,
                "mcl_row_count": mcl_row_count,
                "lease_document_count": document_count,
                "trace_payload_mode": "full-agent-io",
                "llm_enabled": str(bool(os.getenv("OPENAI_API_KEY"))).lower(),
                "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini") if os.getenv("OPENAI_API_KEY") else "disabled",
            })
        except Exception:
            tracker.enabled = False
            tracker.failure_reason = "MLflow rejected run metadata; check server and experiment permissions"
        tracker_token = _ACTIVE_TRACKER.set(tracker)
        yield tracker
    except BaseException:
        tracker.status = "failed"
        _finish_run(tracker)
        try:
            run_context.__exit__(*sys.exc_info())
        except Exception:
            pass
        _ACTIVE_TRACKER.reset(tracker_token)
        raise
    else:
        _finish_run(tracker)
        try:
            run_context.__exit__(None, None, None)
        except Exception:
            tracker.enabled = False
            tracker.failure_reason = "MLflow could not finish saving the run"
        _ACTIVE_TRACKER.reset(tracker_token)


def _finish_run(tracker: "AnalysisTracker") -> None:
    """Write final aggregate metrics and status without affecting the analysis."""
    try:
        if tracker.metrics:
            tracker.metrics["analysis_duration_seconds"] = round(time.perf_counter() - tracker.started_at, 3)
            mlflow.log_metrics(tracker.metrics)
        mlflow.set_tag("analysis_status", tracker.status)
    except Exception:
        pass


class AnalysisTracker:
    """Collect aggregate analysis metrics for the active MLflow run."""

    def __init__(self) -> None:
        self.enabled = False
        self.run_id: str | None = None
        self.tracking_uri: str = DEFAULT_TRACKING_URI
        self.tracking_url: str = DEFAULT_TRACKING_URI
        self.traces_url: str = DEFAULT_TRACKING_URI
        self.failure_reason: str | None = None
        self.span_failures = 0
        self.started_at = 0.0
        self.status = "completed"
        self.metrics: dict[str, float] = {}

    def log_result(self, *, mcl_rows_checked: int, matched_rows: int, review_rows: int,
                   lease_documents_read: int, lease_documents_failed: int) -> None:
        self.metrics.update({
            "mcl_rows_checked": float(mcl_rows_checked),
            "matched_rows": float(matched_rows),
            "review_rows": float(review_rows),
            "lease_documents_read": float(lease_documents_read),
            "lease_documents_failed": float(lease_documents_failed),
        })


@contextmanager
def node_span(name: str, inputs: dict[str, Any]) -> Iterator[Any]:
    """Create a span and serialize its explicit agent inputs and outputs to MLflow."""
    try:
        span_context = mlflow.start_span(name=name, span_type=SpanType.CHAIN)
        live_span = span_context.__enter__()
        live_span.set_inputs(to_trace_json(inputs))
        span = _SerializableSpan(live_span)
    except Exception:
        tracker = _ACTIVE_TRACKER.get()
        if tracker is not None:
            tracker.span_failures += 1
            tracker.failure_reason = "MLflow could not create one or more trace spans"
        yield _NoopSpan()
        return
    try:
        yield span
    except BaseException:
        # Mark the span failed without exporting the original exception, which
        # may contain query text or excerpts from a source document.
        try:
            span.set_outputs({"status": "failed"})
        except Exception:
            pass
        try:
            safe_error = RuntimeError("Analysis stage failed")
            span_context.__exit__(RuntimeError, safe_error, None)
        except Exception:
            pass
        raise
    else:
        try:
            span_context.__exit__(None, None, None)
        except Exception:
            pass


def to_trace_json(value: Any) -> Any:
    """Convert agent state into JSON-safe trace payloads without dropping fields."""
    if isinstance(value, dict):
        return {str(key): to_trace_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_trace_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return to_trace_json(value.model_dump(mode="json"))
    if hasattr(value, "item"):
        try:
            return to_trace_json(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


class _NoopSpan:
    def set_outputs(self, outputs: Any) -> None:
        pass


class _SerializableSpan:
    def __init__(self, span: Any) -> None:
        self._span = span

    def set_outputs(self, outputs: Any) -> None:
        self._span.set_outputs(to_trace_json(outputs))
