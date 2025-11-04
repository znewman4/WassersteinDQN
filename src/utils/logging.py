# src/utils/logging.py
import logging
import os

_LOGGER_INITIALIZED = False

def get_logger(name="default", log_dir="results/logs"):
    global _LOGGER_INITIALIZED
    if not _LOGGER_INITIALIZED:
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(log_dir, "run.log"), mode="w"),
            ],
        )
        _LOGGER_INITIALIZED = True
    return logging.getLogger(name)
