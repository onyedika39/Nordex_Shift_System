import os
import sys
import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    # extracting the traceback details from the exception information
    _, _, exc_tb = error_detail.exc_info()
    # getting the file name and line number where the error occurred

    
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in file: {file_name} at line: {line_number} with message: {str(error)}"
    logging.error(error_message)  # log the error message
    
    return error_message

class MyException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        # return the error message when the exception is converted to a string
        return self.error_message