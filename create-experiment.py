import os
import mlflow
from mlflow.entities.trace_location import UnityCatalog

mlflow.set_tracking_uri("databricks")
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = "704ef8e21124ffa4"

experiment = mlflow.set_experiment(
    experiment_name="/Users/areebnawab98@gmail.com/MCL Renewal Agent UC",
    trace_location=UnityCatalog(
        catalog_name="workspace",
        schema_name="catalog_assante_mcl_renewal_gold_v1",
    ),
)
print(experiment.experiment_id)