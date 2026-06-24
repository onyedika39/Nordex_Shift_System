import sys
import os
import numpy as np
import pandas as pd
import yaml

from src.logger import config_logger
from src.exception import MyException
from config.constant import target_column, SCHEMA_PATH, FEATURE_ARTIFACT

from src.data.data_ingestion import load_data
from src.data.data_validation import starting_data_validation
from src.data.data_processing import DataProcessor, start_data_processing

logger = config_logger()

class FeatureEngineer:

    def __init__(self, data: pd.DataFrame):
        self.data = data
        logger.info("feature engineering initialized...")

    def engineer_features(self):
        try:
            logger.info("started feature engineering process...")

            self.data['start_time'] = pd.to_datetime(self.data['start_time'])
            self.data['end_time'] = pd.to_datetime(self.data['end_time'])
            self.data['date'] = pd.to_datetime(self.data['date'])

            # fixing the overnight shift and any row where end_time < start_time
            mask = self.data['end_time'] < self.data['start_time']
            self.data.loc[mask, 'end_time'] = self.data.loc[mask, 'end_time'] + pd.Timedelta(days=1)

            # shift duration
            self.data['shift_duration'] = (self.data['end_time'] - self.data['start_time']).dt.total_seconds() / 3600

            # production and downtime ratios
            self.data['defect_rate'] = self.data['defect_count'] / self.data['units_produced'].replace(0, np.nan)
            self.data['downtime_ratio'] = self.data['downtime_minutes'] / (self.data['shift_duration'] * 60)

            # temporal features
            self.data['day_of_week'] = self.data['date'].dt.dayofweek
            self.data['hour_of_day'] = self.data['start_time'].dt.hour

            logger.info('feature engineering process completed successfully...')
            return self.data

        except Exception as e:
            raise MyException(e, sys)

    def drop_unnecessary_columns_and_duplicates(self):
        try:
            with open(SCHEMA_PATH, 'r') as f:
                schema = yaml.safe_load(f)
            columns_to_drop = schema["columns"]["columns_to_drop"]

            existing_cols = [
                column
                for column in columns_to_drop
                if column in self.data.columns
            ]
            self.data.drop(columns=existing_cols, inplace=True)

            processor = DataProcessor(self.data)
            self.data = processor.removing_duplicates()

            logger.info("dropped unnecessary columns and removed duplicates...")
            return self.data

        except Exception as e:
            raise MyException(e, sys)

    def feature_engine(self):
        try:
            self.data = self.engineer_features()
            self.data = self.drop_unnecessary_columns_and_duplicates()
            return self.data
        except Exception as e:
            raise MyException(e, sys)


def start_feature_engineering(data: pd.DataFrame):
    try:
        engineer = FeatureEngineer(data)
        data = engineer.feature_engine()

        processor = DataProcessor(data)
        X_train, X_test, y_train, y_test = processor.train_test_splitting(data)

        os.makedirs(os.path.dirname(FEATURE_ARTIFACT), exist_ok=True)
        data.to_csv(FEATURE_ARTIFACT, index=False)

        return X_train, X_test, y_train, y_test

    except Exception as e:
        raise MyException(e, sys)


if __name__ == '__main__':
    raw_data = load_data()
    validated_data = starting_data_validation(raw_data)
    processed_data = start_data_processing(validated_data)
    X_train, X_test, y_train, y_test = start_feature_engineering(processed_data)

    print(X_train.head())
    print(X_train.shape)
    print(y_train.head())