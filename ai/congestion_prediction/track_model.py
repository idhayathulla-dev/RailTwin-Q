import os
import joblib
import pandas as pd
import numpy as np
from ai.congestion_prediction.utils import pred_logger
from ai.congestion_prediction.preprocessing import DataPreprocessor

class TrackCongestionModel:
    def __init__(self, models_dir="models/congestion_predictor"):
        self.models_dir = models_dir
        self.preprocessor = DataPreprocessor(level_name="track")
        self.ensemble_15 = None
        self.ensemble_30 = None
        self.ensemble_60 = None
        self.initialized = False

    def load_model(self):
        """
        Loads preprocessor and ensembles.
        """
        try:
            self.preprocessor.load_preprocessor(self.models_dir)
            self.ensemble_15 = joblib.load(os.path.join(self.models_dir, "track_model_15.pkl"))
            self.ensemble_30 = joblib.load(os.path.join(self.models_dir, "track_model_30.pkl"))
            self.ensemble_60 = joblib.load(os.path.join(self.models_dir, "track_model_60.pkl"))
            self.initialized = True
        except Exception as e:
            pred_logger.error(f"Failed to load Track Congestion Model: {e}")
            self.initialized = False

    def predict_congestion(self, df_tick: pd.DataFrame) -> list:
        """
        Predicts future track occupancy/congestion and calculates uncertainty metrics.
        """
        if not self.initialized:
            self.load_model()
            if not self.initialized:
                return []

        # Preprocess features
        X_processed = self.preprocessor.transform(df_tick)

        # Ensembles predict
        preds_15 = [est.predict(X_processed) for est in self.ensemble_15]
        preds_30 = [est.predict(X_processed) for est in self.ensemble_30]
        preds_60 = [est.predict(X_processed) for est in self.ensemble_60]

        # Calculate statistics
        p_15_mean = np.mean(preds_15, axis=0)
        p_30_mean = np.mean(preds_30, axis=0)
        p_60_mean = np.mean(preds_60, axis=0)

        p_30_std = np.std(preds_30, axis=0)

        results = []
        for idx in range(len(df_tick)):
            track_id = int(df_tick.iloc[idx]["track_id"])
            o_15 = max(0.0, min(100.0, float(p_15_mean[idx])))
            o_30 = max(0.0, min(100.0, float(p_30_mean[idx])))
            o_60 = max(0.0, min(100.0, float(p_60_mean[idx])))
            
            # Estimate track congestion as equal to occupancy index for this simple segment
            c_30 = o_30
            
            # Confidence score & 95% Prediction Interval (mean +/- 1.96 * std)
            std_val = float(p_30_std[idx])
            confidence = max(0.40, min(0.98, np.exp(-std_val / 8.0)))
            
            lower_bound = max(0.0, o_30 - 1.96 * std_val)
            upper_bound = min(100.0, o_30 + 1.96 * std_val)

            results.append({
                "track_id": track_id,
                "occupancy_15": round(o_15, 2),
                "occupancy_30": round(o_30, 2),
                "occupancy_60": round(o_60, 2),
                "congestion_30": round(c_30, 2),
                "confidence": round(confidence, 2),
                "prediction_interval": [round(lower_bound, 2), round(upper_bound, 2)]
            })
            
        return results
