import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(".env"))

import mlflow
from mlflow.genai.datasets import create_dataset, get_dataset

experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "MCL Renewal Agent")
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    raise RuntimeError(
        f"Experiment {experiment_name!r} not found. Set MLFLOW_EXPERIMENT_NAME "
        "to its Databricks workspace path, then rerun."
    )

data_path = Path("evaluation/assante_mcl_renewal_eval.jsonl")
records = [
    json.loads(line)
    for line in data_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

#dataset = create_dataset(
    #name=os.environ["MLFLOW_EVAL_DATASET_NAME"],
    #experiment_id=experiment.experiment_id,
#)
dataset = get_dataset(
    name=os.environ["MLFLOW_EVAL_DATASET_NAME"],
    #experiment_id=experiment.experiment_id,
)
dataset.merge_records(records)
print(f"Registered {len(records)} record(s): {dataset.name} ({dataset.dataset_id})")