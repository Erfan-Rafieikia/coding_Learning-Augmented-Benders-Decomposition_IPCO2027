import logging
import os

def setup_logging(log_file_path="logs/app.log", level=logging.DEBUG):
    # Create directory if not exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Define a formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler: saves all logs
    file_handler = logging.FileHandler(log_file_path, mode='w')  # use 'a' to append instead of overwrite
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Stream handler: console logs
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Get the root logger and add both handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
