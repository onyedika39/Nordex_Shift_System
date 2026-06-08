import pandas as pd
import sqlite3
import sys

from config.constant import database_path
from src.logger import config_logger
from src.exception import MyException

logger = config_logger()

def load_data():
    """
    Load data from the SQLite database file and
    return a pandas DataFrame for use by other pipeline stages.
    """
    try:
        logger.info("Data ingestion initiated...")
        logger.info("Loading data from database file...")
        connection = sqlite3.connect(database_path)
        Shift_Data = pd.read_sql("SELECT * FROM ShiftPerformance", connection)
        connection.close()
        logger.info(Shift_Data.head())
        logger.info("Data loading completed successfully...")
        return Shift_Data
    except Exception as e:
        logger.error(f"Error occurred while loading data: {e}")
        raise MyException(e, sys)

# load_data()
