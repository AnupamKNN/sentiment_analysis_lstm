import sys
from src.sentiment_analysis.logging.logger import logging

class SentimentAnalysisException(Exception):
    """
    Custom exception class for Sentiment Analysis application.
    
    This class extends the
    built-in Exception class to provide a more specific exception
    for the Sentiment Analysis application. It can be used to
    handle errors related to data processing, model training, and
    prediction tasks.
    """

    def __init__(self, error_message, error_details: sys):
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.filename, self.lineno, self.error_message
        )