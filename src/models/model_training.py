import sys
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.logger import config_logger
from src.exception import MyException
from config.constant import SCHEMA_PATH
from src.data.data_ingestion import load_data
from src.data.data_validation import starting_data_validation
from src.data.data_processing import start_data_processing
from src.features.feature_engineering import start_feature_engineering
from src.models.model_pusher import ModelPusher

logger = config_logger()


class ModelTraining:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.squeeze()
        self.y_test = y_test.squeeze()
        with open(SCHEMA_PATH, "r") as f:
            self.schema = yaml.safe_load(f)
        self.model_pipeline = None

    def build_pipeline(self):
        """build pipeline and include the preprocessing method for categorical columns"""

        Numerical_columns = self.schema["columns"]["numerical_features"]
        Categorical_columns = self.schema["columns"]["categorical_features"]

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    Categorical_columns
                ),
                (
                    "num",
                    "passthrough",
                    Numerical_columns
                )
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("pre_process", preprocess),
                ("ml_model", GradientBoostingRegressor())
            ]
        )
        return pipeline

    def train_model(self):
        try:
            logger.info("starting the model training pipeline...")
            self.model_pipeline = self.build_pipeline()
            self.model_pipeline.fit(self.X_train, self.y_train)
            logger.info("Model training successful.")
            return self.model_pipeline

        except Exception as e:
            logger.error(f"Error occurred during model training {e}")
            raise MyException(e, sys)

    def evaluate_model(self):
        try:
            logger.info("evaluating the model performance...")
            y_pred = self.model_pipeline.predict(self.X_test)

            r2 = r2_score(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)

            logger.info(
                f"pipeline evaluation done with the performance of {r2} r2score, {mae} mae score and {mse} mse score"
            )
            return r2, mae, mse

        except Exception as e:
            logger.error(f"error occurred while evaluating the model {e}")
            raise MyException(e, sys)

def start_model_training():
    try:
        raw_data = load_data()
        validated_data = starting_data_validation(raw_data)
        processed_data = start_data_processing(validated_data)

        X_train, X_test, y_train, y_test = start_feature_engineering(processed_data)

        trainer = ModelTraining(X_train, X_test, y_train, y_test)
        pipeline = trainer.train_model()
        r2, mae, mse = trainer.evaluate_model()

        model_pusher = ModelPusher()
        model_was_registered = model_pusher.push_model(
            model=pipeline,
            r2_score=r2,
            mae_score=mae
        )

        if model_was_registered:
            logger.info("New Pipeline registered and promoted in mlflow...")
        else:
            logger.info("Existing model performs better than the new model...")

        return r2, mae, pipeline

    except Exception as e:
        logger.error('error occurred while starting the model training.')
        raise MyException(e, sys)


if __name__ == '__main__':
    start_model_training()