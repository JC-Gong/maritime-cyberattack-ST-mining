import logging
import sys
import os

def setup_logger(name="MC-STM", level=logging.INFO):
    """Set up a standard logger for the project."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_dir = "output"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    fh = logging.FileHandler(os.path.join(log_dir, "analysis.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
