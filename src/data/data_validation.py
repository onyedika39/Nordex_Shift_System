import pandas as pd
import numpy as np
import sys
from src.logger import config_logger
from src.exception import MyException
from src.data.data_ingestion import load_data

logger = config_logger()

class Datavalidator:
    """
    a class that basically checks and validates our data, to know,
    how of quality the data is and if there is any missing values, or so on.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        logger.info("Data validation initiated...")

    def check_data_if_empty(self):
        """
        This function checks if the data is empty or not, and if it is empty, it raises an exception.
        """
        try:
            logger.info("Checking if data is empty...")
            if self.data.empty:
                logger.error("Data is empty.")
                raise MyException("Data is empty", sys)
            else:
                logger.info("Data is not empty.")
        except Exception as e:
            logger.error(f"Error occurred while checking if data is empty: {e}")
            raise MyException(e, sys)

    def check_missing_values(self):
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Missing values detected:\n{missing_values}")
        else:
            logger.info("No missing values detected.")
        return missing_values

    def checking_duplicates(self):
        """
        this function checks if the data has duplicates
        """
        try:
            duplicates = self.data.duplicated().sum()
            if duplicates > 0:
                logger.warning(f"Duplicate records detected: {duplicates}")
            else:
                logger.info("No duplicate records detected.")
        except Exception as e:
            logger.error(f"Error occurred while checking for duplicates: {e}")
            raise MyException(e, sys)


def starting_data_validation(data: pd.DataFrame):
    try:
        logger.info("the data validator pipeline has started...")
        validator_engine = Datavalidator(data)
        validator_engine.check_data_if_empty()
        validator_engine.check_missing_values()
        validator_engine.checking_duplicates()

        logger.info("Data validation completed successfully.")
        logger.info(data.head())
        return data
    except Exception as e:
        logger.error(f"Error occurred during data validation: {e}")
        raise MyException(e, sys)

data = load_data()
data = starting_data_validation(data)