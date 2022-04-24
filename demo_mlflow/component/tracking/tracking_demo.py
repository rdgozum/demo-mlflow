import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import pandas as pd
from pathlib import Path

EXPERIMENT_NAME = "tracking_demo"
TRACKING_PATH = Path().resolve().parent / "demo_mlflow/mlruns"


class TrackingDemo:
    def initialize_experiment(self):
        mlflow.set_tracking_uri("file://" + TRACKING_PATH.as_posix())

        try:
            self.experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        except MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(
                EXPERIMENT_NAME
            ).experiment_id
            # mlflow.set_experiment(self.experiment_id)
        self.runs = []

    def log(self, run_name, **items_dict):
        experiment_id = self.experiment_id

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            self.runs.append(run.info.run_id)

            # Track parameters
            for key in items_dict["parameters"]:
                mlflow.log_param(key, items_dict["parameters"][key])

            # Track metrics
            for key in items_dict["metrics"]:
                mlflow.log_metric(key, items_dict["metrics"][key])

            # Track model
            for key in items_dict["models"]:
                mlflow.sklearn.log_model(items_dict["models"][key], "classifier")

    def query(self):
        client = MlflowClient()

        # Retrieve Experiment information
        experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

        # Retrieve runs information (parameter 'depth', 'metric', 'accuracy')
        all_run_infos = client.list_run_infos(experiment_id)
        all_runs_id = [run.run_id for run in all_run_infos]
        all_params = [
            client.get_run(run_id).data.params["depth"] for run_id in all_runs_id
        ]
        all_metrics = [
            client.get_run(run_id).data.metrics["accuracy"] for run_id in all_runs_id
        ]

        # View runs information
        df = pd.DataFrame(
            {"run_ids": all_runs_id, "parameters": all_params, "metrics": all_metrics}
        )

        # Retrieve artifact from best run
        best_run_id = df.sort_values("metrics", ascending=False).iloc[0]["run_ids"]
        best_model_path = client.download_artifacts(best_run_id, "classifier")
        best_model = mlflow.sklearn.load_model(best_model_path)

        print(df.head())
        print(type(best_model))
        print(best_model.get_params())

        # Delete runs
        # for run_id in all_runs_id:
        #    client.delete_run(run_id)

        # Delete experiment
        # client.delete_experiment(experiment_id)
