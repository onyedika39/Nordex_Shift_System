import dagshub
import mlflow
from config.constant import EXPERIMENT_NAME

def setup_mlflow():
    dagshub.init(
        repo_owner="onyedikakenechukwu7",
        repo_name="Nordex_Shift_System",
        mlflow=True
    )
    mlflow.set_experiment(EXPERIMENT_NAME)