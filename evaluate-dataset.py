"""Run the LangGraph renewal agent against a registered MLflow GenAI dataset."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv(".env")

import mlflow
from mlflow.entities.trace_location import UnityCatalog
from mlflow.genai.datasets import get_dataset
from mlflow.genai.scorers import Correctness, ExpectationsGuidelines

from mcl_agent.graph import build_agent_graph


DATASET_NAME = os.getenv(
    "MLFLOW_EVAL_DATASET_NAME",
    "workspace.catalog_assante_mcl_renewal_gold_v1.mcl_renewal_eval",
)
TRACE_CATALOG = os.getenv("MLFLOW_TRACE_CATALOG", "workspace")
TRACE_SCHEMA = os.getenv(
    "MLFLOW_TRACE_SCHEMA", "catalog_assante_mcl_renewal_gold_v1"
)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "MCL Renewal Agent")


def configure_evaluation_experiment() -> str:
    """Check the experiment and ensure traces use the requested UC schema."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
    mlflow.set_tracking_uri(tracking_uri)

    warehouse_id = os.getenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", "704ef8e21124ffa4")

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment {EXPERIMENT_NAME!r} was not found in {tracking_uri!r}."
        )

    desired_location = UnityCatalog(
        catalog_name=TRACE_CATALOG,
        schema_name=TRACE_SCHEMA,
    )
    existing_location = experiment.trace_location
    if existing_location is not None:
        existing_catalog = getattr(existing_location, "catalog_name", None)
        existing_schema = getattr(existing_location, "schema_name", None)
        if (existing_catalog, existing_schema) != (TRACE_CATALOG, TRACE_SCHEMA):
            raise RuntimeError(
                "The experiment already has a different trace location; refusing to "
                "change it. Use an experiment bound to the intended Unity Catalog schema."
            )
    else:
        if os.getenv("MLFLOW_ALLOW_PERMANENT_UC_TRACE_BINDING", "").lower() != "true":
            raise RuntimeError(
                f"Experiment {EXPERIMENT_NAME!r} has no Unity Catalog trace location. "
                f"Binding it to {TRACE_CATALOG}.{TRACE_SCHEMA} is permanent. After "
                "reviewing this change, set MLFLOW_ALLOW_PERMANENT_UC_TRACE_BINDING=true "
                "to authorize the binding and rerun."
            )
        os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = warehouse_id
        experiment = mlflow.set_experiment(
            experiment_id=experiment.experiment_id,
            trace_location=desired_location,
        )

    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = warehouse_id
    return experiment.experiment_id


def main() -> None:
    experiment_id = configure_evaluation_experiment()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for the OpenAI-powered correctness and "
            "guidelines judges used by this evaluation."
        )

    try:
        mlflow.langchain.autolog(log_traces=True, silent=True)
    except Exception as exc:
        raise RuntimeError("Could not enable MLflow tracing for LangGraph.") from exc

    dataset = get_dataset(name=DATASET_NAME)
    graph = build_agent_graph()

    @mlflow.trace(name="mcl_renewal_agent_predict")
    def predict_fn(query: str, mcl_rows: list[dict], lease_documents: list[dict]) -> str:
        result = graph.invoke(
            {
                "query": query,
                "mcl_rows": mcl_rows,
                "lease_documents": lease_documents,
            }
        )
        return result["answer"]

    judge_model = "openai:/gpt-6-luna"
    result = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=predict_fn,
        scorers=[
            Correctness(
                model=judge_model,
                inference_params={"temperature": 1},
            ),
            ExpectationsGuidelines(
                model=judge_model,
                inference_params={"temperature": 1},
            ),
        ],
    )
    mlflow.flush_trace_async_logging()
    print(f"Dataset: {dataset.name}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Evaluation results: {result}")


if __name__ == "__main__":
    main()
