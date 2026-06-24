import sys
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.logger import config_logger
from src.exception import MyException
from src.utils.mlflow_setup import setup_mlflow
from config.constant import registered_model_name

logger = config_logger()


def get_best_existing_model_metrics(model_name: str):
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(model_name)

        if not versions:
            return None, None

        best_r2, best_mae = None, None
        for v in versions:
            run_id = v.run_id
            run = client.get_run(run_id)
            r2 = run.data.metrics.get("r2_score")
            mae = run.data.metrics.get("mae_score")
            if r2 is None or mae is None:
                continue
            if (
                best_r2 is None
                or (r2 > best_r2)
                or (r2 == best_r2 and mae < best_mae)
            ):
                best_r2 = r2
                best_mae = mae

        return best_r2, best_mae

    except Exception:
        logger.warning("No existing model or metrics found")
        return None, None


def load_model_from_mlflow(model_name=registered_model_name):
    try:
        setup_mlflow()
        client = MlflowClient()
        logger.info(f"fetching the latest model: {model_name}")

        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            logger.warning(f"No model versions found for {model_name}")
            return None

        latest_version = max(versions, key=lambda v: int(v.version))
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info(f"model version {latest_version.version} loaded successfully")
        return model

    except Exception as e:
        logger.error(f"error occurred while loading the model from mlflow {e}")
        raise MyException(e, sys)
