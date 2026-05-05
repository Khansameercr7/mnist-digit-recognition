"""
Logging utilities for MNIST project
- Consistent logging setup across all modules
- Logs to file and console with timestamps
"""

import logging
import os
from datetime import datetime

import config


def setup_logger(name, log_level=logging.INFO):
    """
    Create and configure a logger with both file and console handlers.
    
    Args:
        name (str): Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Don't add handlers if already added
    if logger.hasHandlers():
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (daily log file)
    log_filename = os.path.join(
        config.LOGS_DIR,
        f"mnist_{datetime.now().strftime('%Y-%m-%d')}.log"
    )
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name):
    """
    Get an existing logger or create a new one.
    
    Args:
        name (str): Logger name (usually __name__)
        
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger = setup_logger(name)
    return logger
