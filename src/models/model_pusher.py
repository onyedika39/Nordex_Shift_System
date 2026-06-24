import sys
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.logger import config_logger
from src.exception import MyException
from src.utils.mlflow_setup import setup_mlflow
from src.utils.model_utils import get_best_existing_model_metrics
from config.constant import registered_model_name

logger = config_logger()


class ModelPusher:
    def __init__(self):
        setup_mlflow()
        self.client = MlflowClient()
        self.registered_model_name = registered_model_name

    def push_model(self, model, r2_score: float, mae_score: float) -> bool:
        """
        Push model to MLflow registry if it performs better than the existing model.
        """
        try:
            logger.info("Checking the existing model performance...")
            old_r2, old_mae = get_best_existing_model_metrics(self.registered_model_name)
            push_new_model = False

            if old_r2 is None:
                push_new_model = True
                logger.info("No existing model found, registering new model...")
            elif r2_score > old_r2:
                push_new_model = True
                logger.info("New model performed better than the old one")
            elif r2_score == old_r2 and mae_score < old_mae:
                push_new_model = True

            if push_new_model:
                with mlflow.start_run():
                    logger.info("Logging new model to mlflow...")
                    mlflow.log_metric("r2_score", r2_score)
                    mlflow.log_metric("mae_score", mae_score)
                    model_info = mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name=self.registered_model_name
                    )

                logger.info(f"Model registered: {model_info.model_uri}")
                return True
            else:
                logger.info("Existing model performs better, skipping registration.")
                return False

        except Exception as e:
            logger.error(f"error occurred while pushing the model: {e}")
            raise MyException(e, sys)
