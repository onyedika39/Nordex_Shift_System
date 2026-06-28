from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
import datetime
import optuna
import sys

from src.logger import config_logger
from src.exception import MyException
from src.pipeline.prediction import prediction_pipeline
from src.pipeline.training import start_model_training
from src.utils.model_utils import load_model_from_mlflow


logger = config_logger()

app = FastAPI(
    title="Nordex Shift Performance"
)

model = None


@app.on_event("startup")
def load_model_on_startup():
    global model

    try:
        logger.info("loading model at startup...")

        model = load_model_from_mlflow()

        if model is not None:
            logger.info("model successfully loaded...")
        else:
            logger.warning("No registered model found. Run /retrain to train and register a model.")

    except Exception as e:
        logger.warning(f"Could not load model from mlflow: {e}. Run /retrain to train and register a model.")


class ShiftInput(BaseModel):
    units_produced: int
    defect_count: int
    cycle_time_avg: float
    experience_level: int
    runtime_hours: float
    downtime_minutes: float
    maintenance_flag: int
    maintenance_downtime: float
    temperature: float
    humidity: float
    shift_duration: float
    day_of_week: int
    hour_of_day: int = datetime.datetime.now().hour
    shift_name: str
    skill_category: str
    machine_status: str
    issue_type: str = "No Issue"
    defect_type: str = "None"
    severity: str = "Low"


class OptimizationInput(BaseModel):
    exp_range: list
    downtime_range: list
    defect_range: list
    n_trials: int = 100


@app.get("/")
def health_check():
    return {
        "message": "Nordex shift performance API is Running"
    }


@app.post("/predict")
def predict_shift_efficiency(input: ShiftInput):
    try:
        if model is None:
            raise Exception("model is not loaded from mlflow...")

        result = prediction_pipeline(
            input.dict(),
            model
        )

        return {
            "predicted_shift_efficiency_score": float(result[0])
        }

    except Exception as e:
        logger.error("prediction failed")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


def retrain_pipeline():
    global model

    try:
        start_model_training()

        logger.info("Reloading the latest model from mlflow...")

        model = load_model_from_mlflow()

    except Exception as e:
        logger.error("re-training process failed...")
        raise MyException(e, sys)


@app.post("/retrain")
def retrain_model():
    thread = threading.Thread(
        target=retrain_pipeline
    )

    thread.start()

    return {
        "message": "Training started in the background, model will automatically update."
    }


@app.post("/optimize")
def optimize_shift(input: OptimizationInput):
    try:
        if model is None:
            raise Exception("model not loaded from mlflow")

        def objective(trial):
            experience_level = trial.suggest_int(
                "experience_level",
                input.exp_range[0],
                input.exp_range[1]
            )

            downtime_minutes = trial.suggest_int(
                "downtime_minutes",
                input.downtime_range[0],
                input.downtime_range[1]
            )

            defect_count = trial.suggest_int(
                "defect_count",
                input.defect_range[0],
                input.defect_range[1]
            )

            maintenance_downtime = trial.suggest_int(
                "maintenance_downtime",
                0,
                120
            )

            units_produced = trial.suggest_int(
                "units_produced",
                600,
                1200
            )

            maintenance_flag = trial.suggest_int(
                "maintenance_flag",
                0,
                1
            )

            cycle_time_avg = trial.suggest_int(
                "cycle_time_avg",
                30,
                40
            )

            temperature = trial.suggest_int(
                "temperature",
                18,
                30
            )

            humidity = trial.suggest_int(
                "humidity",
                30,
                70
            )

            runtime_hours = 7.5
            shift_duration = runtime_hours
            day_of_week = datetime.datetime.today().weekday()
            hour_of_day = datetime.datetime.now().hour

            data = {
                "units_produced": units_produced,
                "defect_count": defect_count,
                "cycle_time_avg": cycle_time_avg,
                "experience_level": experience_level,
                "runtime_hours": runtime_hours,
                "downtime_minutes": downtime_minutes,
                "maintenance_flag": maintenance_flag,
                "maintenance_downtime": maintenance_downtime,
                "temperature": temperature,
                "humidity": humidity,
                "shift_duration": shift_duration,
                "day_of_week": day_of_week,
                "hour_of_day": hour_of_day,
                "shift_name": "Morning",
                "skill_category": "Medium",
                "machine_status": "Running",
                "issue_type": "No Issue",
                "defect_type": "None",
                "severity": "Low"
            }

            prediction = prediction_pipeline(
                data,
                model
            )

            return prediction[0]

        study = optuna.create_study(
            direction="maximize"
        )

        study.optimize(
            objective,
            n_trials=input.n_trials
        )

        return {
            "best_score": study.best_value,
            "best_parameters": study.best_params
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
