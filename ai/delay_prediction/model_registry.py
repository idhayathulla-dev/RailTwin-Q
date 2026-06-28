import os
import json
import datetime
from ai.delay_prediction.utils import train_logger

class ModelRegistry:
    @staticmethod
    def register_model(
        models_dir: str,
        version: str,
        feature_count: int,
        metrics: dict,
        hyperparameters: dict,
        dataset_info: str = "synthetic_datasets_v2"
    ):
        """
        Appends or updates model training metrics, parameters, and metadata inside model_registry.json.
        """
        registry_path = os.path.join(models_dir, "model_registry.json")
        os.makedirs(models_dir, exist_ok=True)
        
        # Load existing registry
        registry = {}
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
            except Exception as e:
                train_logger.warning(f"Error loading model registry file: {e}. Resetting registry.")
                registry = {}

        # Log entry
        entry = {
            "version": version,
            "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_version": dataset_info,
            "feature_count": feature_count,
            "evaluation_metrics": metrics,
            "hyperparameters": hyperparameters
        }

        registry[version] = entry

        # Save registry
        try:
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=4)
            train_logger.info(f"Successfully registered model version {version} in registry: {registry_path}")
        except Exception as e:
            train_logger.error(f"Error writing model registry file: {e}")
            
        return entry
