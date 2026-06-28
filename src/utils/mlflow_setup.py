import dagshub
import mlflow
import os
from dotenv import load_dotenv
from config.constant import EXPERIMENT_NAME, repo_owner, repo_name

load_dotenv(override=True)


def setup_mlflow():
    dagshub_token = os.getenv("MLFLOW_TOKEN")
    tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME", repo_owner)
    tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD", dagshub_token)

    if tracking_password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = tracking_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = tracking_password
        mlflow.set_tracking_uri(
            f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        )
    else:
        dagshub.init(
            repo_owner=repo_owner,
            repo_name=repo_name,
            mlflow=True
        )

    mlflow.set_experiment(EXPERIMENT_NAME)
