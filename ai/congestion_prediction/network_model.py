import os
import joblib
import pandas as pd
import numpy as np
from ai.congestion_prediction.utils import pred_logger
from ai.congestion_prediction.preprocessing import DataPreprocessor

class NetworkCongestionModel:
    def __init__(self, models_dir="models/congestion_predictor"):
        self.models_dir = models_dir
        self.preprocessor = DataPreprocessor(level_name="network")
        self.ensembles = {}
        self.initialized = False

    def load_model(self):
        """
        Loads preprocessor and K-Fold ensembles for all targets.
        """
        try:
            self.preprocessor.load_preprocessor(self.models_dir)
            targets = ["network_congestion", "platform_utilization", "track_utilization", "average_delay"]
            for target in targets:
                for horizon in [15, 30, 60]:
                    key = f"{target}_{horizon}"
                    filename = f"network_model_{key}.pkl"
                    self.ensembles[key] = joblib.load(os.path.join(self.models_dir, filename))
            self.initialized = True
        except Exception as e:
            pred_logger.error(f"Failed to load Network Congestion Model: {e}")
            self.initialized = False

    def predict_congestion(self, df_tick: pd.DataFrame) -> dict:
        """
        Predicts future global network congestion targets.
        """
        if not self.initialized:
            self.load_model()
            if not self.initialized:
                return {}

        # Preprocess features
        X_processed = self.preprocessor.transform(df_tick)

        predictions = {}
        stds = {}
        targets = ["network_congestion", "platform_utilization", "track_utilization", "average_delay"]

        for target in targets:
            for horizon in [15, 30, 60]:
                key = f"{target}_{horizon}"
                ensemble = self.ensembles[key]
                preds = [est.predict(X_processed) for est in ensemble]
                predictions[key] = max(0.0, float(np.mean(preds)))
                if horizon == 30:
                    stds[target] = float(np.std(preds))

        # Compile Stress Index: composite of average predicted congestion and utilization targets
        stress_15 = (predictions["network_congestion_15"] + predictions["platform_utilization_15"] + predictions["track_utilization_15"]) / 3.0
        stress_30 = (predictions["network_congestion_30"] + predictions["platform_utilization_30"] + predictions["track_utilization_30"]) / 3.0
        stress_60 = (predictions["network_congestion_60"] + predictions["platform_utilization_60"] + predictions["track_utilization_60"]) / 3.0

        # Confidence score (exponential decay of standard deviation across network congestion predictions)
        std_val = stds.get("network_congestion", 0.0)
        confidence = max(0.40, min(0.98, np.exp(-std_val / 8.0)))
        
        # 95% Prediction Interval for +30m network congestion
        lower_bound = max(0.0, predictions["network_congestion_30"] - 1.96 * std_val)
        upper_bound = min(100.0, predictions["network_congestion_30"] + 1.96 * std_val)

        return {
            "network_congestion_15": round(predictions["network_congestion_15"], 2),
            "network_congestion_30": round(predictions["network_congestion_30"], 2),
            "network_congestion_60": round(predictions["network_congestion_60"], 2),
            
            "platform_utilization_15": round(predictions["platform_utilization_15"], 2),
            "platform_utilization_30": round(predictions["platform_utilization_30"], 2),
            "platform_utilization_60": round(predictions["platform_utilization_60"], 2),
            
            "track_utilization_15": round(predictions["track_utilization_15"], 2),
            "track_utilization_30": round(predictions["track_utilization_30"], 2),
            "track_utilization_60": round(predictions["track_utilization_60"], 2),
            
            "average_delay_15": round(predictions["average_delay_15"], 2),
            "average_delay_30": round(predictions["average_delay_30"], 2),
            "average_delay_60": round(predictions["average_delay_60"], 2),
            
            "stress_index_15": round(stress_15, 2),
            "stress_index_30": round(stress_30, 2),
            "stress_index_60": round(stress_60, 2),
            
            "confidence": round(confidence, 2),
            "prediction_interval": [round(lower_bound, 2), round(upper_bound, 2)]
        }
