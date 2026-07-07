import os
import logging

def get_logger(name: str, log_file_name: str) -> logging.Logger:
    """
    Configures and returns a logger that writes to console and a specific log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist to prevent duplicate logging
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        log_dir = "."
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, log_file_name)
        
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

# Preconfigured loggers
train_logger = get_logger("Congestion_Training", "congestion_training.log")
pred_logger = get_logger("Congestion_Prediction", "congestion_prediction.log")
eval_logger = get_logger("Congestion_Evaluation", "congestion_evaluation.log")
