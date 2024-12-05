import logging
import os
from datetime import datetime

def setup_logger(name='albert_trading_bot'):
    """
    Configure comprehensive logging for the trading bot
    
    Args:
        name (str): Name of the logger
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    
    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create global logger instance
logger = setup_logger()

def log_trade_event(event_type, details):
    """
    Log specific trading events with structured information
    
    Args:
        event_type (str): Type of trading event
        details (dict): Event details
    """
    log_message = f"Event: {event_type}\n"
    for key, value in details.items():
        log_message += f"  {key}: {value}\n"
    
    logger.info(log_message)

def log_error(error_message, exception=None):
    """
    Log error with optional exception details
    
    Args:
        error_message (str): Error description
        exception (Exception, optional): Exception object
    """
    logger.error(error_message)
    if exception:
        logger.exception(exception)

# Example usage in other modules
# from src.logger import logger, log_trade_event, log_error
