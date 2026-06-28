import os
import joblib
import pandas as pd
import numpy as np
from ai.delay_prediction.utils import pred_logger
from ai.delay_prediction.feature_engineering import FeatureEngineer

class InferenceEngine:
    def __init__(self, models_dir="models/delay_predictor"):
        self.models_dir = models_dir
        self.preprocessor = None
        self.ensemble_15 = None
        self.ensemble_30 = None
        self.ensemble_60 = None
        self.feature_importances = None
        self.initialized = False

    def initialize(self):
        """
        Loads models and preprocessors from disk.
        """
        try:
            pred_logger.info("Initializing Delay Prediction Inference Engine...")
            
            # Preprocessor loader
            from ai.delay_prediction.preprocessing import DataPreprocessor
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load_preprocessors(self.models_dir)

            # Load ensembles (lists of K-fold estimators)
            self.ensemble_15 = joblib.load(os.path.join(self.models_dir, "delay_15min.pkl"))
            self.ensemble_30 = joblib.load(os.path.join(self.models_dir, "delay_30min.pkl"))
            self.ensemble_60 = joblib.load(os.path.join(self.models_dir, "delay_60min.pkl"))

            # Load feature importances from first model for explainable factor extraction
            if hasattr(self.ensemble_30[0], "feature_importances_"):
                self.feature_importances = self.ensemble_30[0].feature_importances_
            else:
                self.feature_importances = np.ones(len(self.preprocessor.feature_ordering))

            self.initialized = True
            pred_logger.info("Inference Engine loaded successfully.")
        except Exception as e:
            pred_logger.error(f"Failed to initialize Inference Engine: {e}")
            self.initialized = False

    def predict_train_delays(self, df_tick: pd.DataFrame) -> list:
        """
        Runs predictions using K-Fold ensembles, calculates statistical variance-based
        confidence, and derives local SHAP-like explainable factors.
        """
        if not self.initialized:
            self.initialize()
            if not self.initialized:
                pred_logger.warning("Inference engine not initialized. Returning empty predictions.")
                return []

        if df_tick.empty:
            return []

        # 1. Feature Engineering
        df_engineered = FeatureEngineer.engineer_features(df_tick)

        # 2. Transform Features
        X_processed = self.preprocessor.transform(df_engineered)

        # 3. Predict using all estimators in K-Fold ensemble
        preds_15_list = [est.predict(X_processed) for est in self.ensemble_15]
        preds_30_list = [est.predict(X_processed) for est in self.ensemble_30]
        preds_60_list = [est.predict(X_processed) for est in self.ensemble_60]

        # Calculate prediction means
        p_15_means = np.mean(preds_15_list, axis=0)
        p_30_means = np.mean(preds_30_list, axis=0)
        p_60_means = np.mean(preds_60_list, axis=0)

        # Calculate prediction standard deviations (ensemble variance)
        p_30_stds = np.std(preds_30_list, axis=0)

        # Factor mapping dictionary for judges
        factor_mapping = {
            "current_delay": "Current Delay",
            "track_occupancy": "Track Occupancy",
            "station_occupancy": "Station Occupancy",
            "rain_intensity": "Heavy Rain",
            "signal_failure": "Signal Failure",
            "maintenance": "Track Maintenance",
            "track_blockage": "Track Blockage",
            "average_network_delay": "Network Delay",
            "network_congestion_score": "Network Congestion",
            "visibility": "Low Visibility",
            "route_complexity_score": "Route Complexity",
            "waiting_trains": "Waiting Trains Queue",
            "platforms_available": "Station Platform Limits",
            "delay_change_last_tick": "Recent Delay Delta",
            "average_station_delay": "Station Congestion Delay"
        }

        results = []
        for idx in range(len(df_tick)):
            train_no = int(df_tick.iloc[idx]["train_no"])
            p_15 = max(0.0, float(p_15_means[idx]))
            p_30 = max(0.0, float(p_30_means[idx]))
            p_60 = max(0.0, float(p_60_means[idx]))

            # Ensemble-based statistical confidence (exponential decay of standard deviation)
            std_val = float(p_30_stds[idx])
            # E.g. std of 0 mins = 98% conf, std of 5 mins = 36% conf (decaying rate)
            confidence = np.exp(-std_val / 6.0)
            confidence = max(0.40, min(0.98, confidence))

            # Local feature attribution (impact = preprocessed value * global importance)
            row_vals = X_processed.iloc[idx].values
            impacts = np.abs(row_vals * self.feature_importances)
            
            # Sort and find top 3 contributing factors
            top_indices = np.argsort(impacts)[::-1][:3]
            top_features = [self.preprocessor.feature_ordering[i] for i in top_indices]
            readable_factors = [factor_mapping.get(f, f.replace("_", " ").title()) for f in top_features]

            results.append({
                "state_id": str(df_tick.iloc[idx].get("state_id", "N/A")),
                "train_id": train_no,
                "delay_predictions": {
                    "15": round(p_15, 2),
                    "30": round(p_30, 2),
                    "60": round(p_60, 2)
                },
                "confidence": round(confidence, 2),
                "top_factors": readable_factors,
                "model_version": "DelayPredictor-v1"
            })

        return results
