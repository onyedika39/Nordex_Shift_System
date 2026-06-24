import pandas as pd
import sys
from src.logger import config_logger
from src.exception import MyException

logger = config_logger()


def prediction_pipeline(input_data: dict, model):
    """
    Uses the preloaded model from MLflow to return a prediction.

    Input:
        input_data: dictionary from the API
        model: preloaded model from MLflow

    Output:
        model prediction as a list
    """
    try:
        logger.info("preparing input data for prediction...")

        df = pd.DataFrame([input_data])

        if "defect_rate" not in df.columns:
            df["defect_rate"] = df["defect_count"] / df["units_produced"].replace(0, float("nan"))

        if "downtime_ratio" not in df.columns:
            df["downtime_ratio"] = df["downtime_minutes"] / (df["shift_duration"] * 60)

        prediction = model.predict(df)

        logger.info("prediction complete")

        return prediction.tolist()

    except Exception as e:
        logger.error(f"error occurred in prediction pipeline {e}")
        raise MyException(e, sys)
