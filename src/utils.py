import os
import logging
from glob import glob
from datetime import datetime
import pytz
import openai
import yaml
import argparse


openai.log = logging.getLogger("openai")
openai.log.setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


class HTTPFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("HTTP")


def get_pacific_time():
    current_time = datetime.now()
    pacific = pytz.timezone("Asia/Seoul")
    pacific_time = current_time.astimezone(pacific)
    return pacific_time


def create_logger(logging_dir, name="log"):
    """
    Create a logger that writes to a log file and stdout.
    """
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    log_file_base = os.path.join(logging_dir, name)
    num = len(glob(log_file_base + "*"))
    log_file_path = log_file_base + "-" + f"{num:03d}" + ".log"
    
    http_filter = HTTPFilter()

    # Create a dedicated logger (not root logger) to avoid conflicts with other libraries
    logger = logging.getLogger("prompt optimization agent")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter("%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    # Add stdout handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(http_filter)
    logger.addHandler(stream_handler)
    
    # Add file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(http_filter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    # Suppress noisy loggers
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("datasets").setLevel(logging.CRITICAL)
    
    return logger

