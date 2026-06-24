import pandas as pd
from src.logger import config_logger
from src.exception import MyException
import sys
from sklearn.model_selection import train_test_split
from config.constant import target_column
from src.data.data_ingestion import load_data
from src.data.data_validation import starting_data_validation

logger = config_logger()

class DataProcessor:

    def __init__(self, data: pd.DataFrame):
        self.data = data
        logger.info("Data Processor initialized...")

    def filling_missing_value(self):
        try:
            logger.info("Filling missing value started...")
            self.data = self.data.sort_values(by="date")

            # Fill temperature & Humidity columns
            self.data["temperature"] = self.data["temperature"].ffill().fillna(
                self.data["temperature"].mean()
            )

            self.data["humidity"] = self.data["humidity"].ffill().fillna(
                self.data["humidity"].mean()
            )

            # Fill Timestamp
            self.data["timestamp"] = self.data["timestamp"].ffill()

            # Fill the categorical fields
            self.data["issue_type"] = self.data["issue_type"].fillna("No Issue")

            self.data["resolved_by"] = self.data["resolved_by"].fillna(
                "No Maintenance"
            )

            # Fill the downtime column
            self.data["maintenance_downtime"] = self.data[
                "maintenance_downtime"
            ].fillna(0)

            self.data = self.data.drop(columns=["maintenance_id"])

            logger.info("Filling missing value completed successfully...")

            return self.data

        except Exception as e:
            logger.error(f"Error occurred while filling missing values: {e}")
            raise MyException(e, sys)

    def removing_duplicates(self):
        try:
            duplicates = self.data.duplicated().sum()
            if duplicates > 0:
                logger.warning(f"Removing {duplicates} duplicated rows")
                self.data = self.data.drop_duplicates()

            return self.data

        except Exception as e:
            raise MyException(e, sys)

    def preprocess_data(self):
        try:
            logger.info("starting the data preprocessing pipeline")

            self.data = self.filling_missing_value()
            self.data = self.removing_duplicates()

            logger.info("data processing completed...")
            return self.data

        except Exception as e:
            logger.error("error occurred while processing the data...")
            raise MyException(e, sys)

    def split_X_y(self, data: pd.DataFrame):
        try:
            X = data.drop(columns=[target_column])

            y = data[target_column]

            logger.info(X.head())
            logger.info(y.head())

            logger.info(
                "data has been successfully splitted into features and target columns..."
            )

            return X, y

        except Exception as e:
            logger.error(
                "error occurred while splitting into features and target columns"
            )
            raise MyException(e, sys)

    def train_test_splitting(self, data: pd.DataFrame):
        try:
            X, y = self.split_X_y(data)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )
            logger.info("Train-test splitting completed...")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(
                "error occurred while splitting the data into training and testing..."
            )
            raise MyException(e, sys)


def start_data_processing(data: pd.DataFrame):
    try:
        processor = DataProcessor(data)
        data = processor.preprocess_data()
        return data

    except Exception as e:
        raise MyException(e, sys)

if __name__ == '__main__':
    data = load_data()
    data = starting_data_validation(data)
    data = start_data_processing(data)
