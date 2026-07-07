import os
import joblib
import numpy as np
import pandas as pd
from ai.congestion_prediction.utils import pred_logger
from ai.congestion_prediction.preprocessing import DataPreprocessor
from ai.congestion_prediction.station_model import StationCongestionModel
from ai.congestion_prediction.track_model import TrackCongestionModel
from ai.congestion_prediction.network_model import NetworkCongestionModel
from ai.congestion_prediction.feature_engineering import FeatureEngineer

class FutureNetworkState:
    """
    Structured object containing the predicted future state of the entire railway network.
    Consumed directly by the Quantum Optimization and Delay Propagation layers.
    """
    def __init__(self, time_horizon: int, tick: int, state_id: str):
        self.time_horizon = time_horizon
        self.tick = tick
        self.state_id = state_id
        self.stations = {}  # station_id -> predicted details
        self.tracks = {}    # track_id -> predicted details
        self.network = {}   # global network predicted statistics
        self.timestamp = pd.Timestamp.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "time_horizon_minutes": self.time_horizon,
            "tick": self.tick,
            "state_id": self.state_id,
            "predictions_timestamp": self.timestamp,
            "predicted_stations": self.stations,
            "predicted_tracks": self.tracks,
            "predicted_network": self.network
        }

class HierarchicalInferenceEngine:
    def __init__(self, models_dir="models/congestion_predictor"):
        self.models_dir = models_dir
        self.station_predictor = StationCongestionModel(models_dir)
        self.track_predictor = TrackCongestionModel(models_dir)
        self.network_predictor = NetworkCongestionModel(models_dir)
        self.initialized = False

    def initialize(self):
        """
        Initializes preprocessors and estimators.
        """
        try:
            pred_logger.info("Initializing Hierarchical Congestion Prediction Inference Engine...")
            self.station_predictor.load_model()
            self.track_predictor.load_model()
            self.network_predictor.load_model()
            self.initialized = True
            pred_logger.info("Hierarchical Inference Engine loaded successfully.")
        except Exception as e:
            pred_logger.error(f"Failed to initialize Hierarchical Inference Engine: {e}")
            self.initialized = False

    def predict_future_railway_state(self, df_station: pd.DataFrame, df_track: pd.DataFrame, df_network: pd.DataFrame) -> dict:
        """
        Runs stacked prediction Level 1 -> Level 2 -> Level 3.
        Returns a dict mapping horizons (15, 30, 60) to FutureNetworkState dict contracts.
        """
        if not self.initialized:
            self.initialize()
            if not self.initialized:
                pred_logger.warning("Engine not initialized. Returning empty states.")
                return {}

        df_st = df_station.copy()
        df_tr = df_track.copy()
        df_net = df_network.copy()

        # Step A: Feature Engineer
        df_st_eng = FeatureEngineer.engineer_features(df_st)
        df_tr_eng = FeatureEngineer.engineer_features(df_tr)
        df_net_eng = FeatureEngineer.engineer_features(df_net)

        # -------------------------------------------------------------
        # LEVEL 1: Station Congestion Predictions
        # -------------------------------------------------------------
        station_predictions = self.station_predictor.predict_congestion(df_st_eng)
        
        # Build map for fast stacked lookups
        lookup_st_30 = {p["station_id"]: p["congestion_30"] for p in station_predictions}
        lookup_st_15 = {p["station_id"]: p["congestion_15"] for p in station_predictions}
        lookup_st_60 = {p["station_id"]: p["congestion_60"] for p in station_predictions}
        
        # Append predicted station congestion directly into Level 2 dataframe inputs (Stacked learning)
        df_tr_eng["src_station_congestion"] = df_tr_eng["source_station_id"].map(lookup_st_30).fillna(0.0)
        df_tr_eng["dest_station_congestion"] = df_tr_eng["destination_station_id"].map(lookup_st_30).fillna(0.0)

        # -------------------------------------------------------------
        # LEVEL 2: Track Congestion Predictions
        # -------------------------------------------------------------
        track_predictions = self.track_predictor.predict_congestion(df_tr_eng)
        
        # Build map for Level 3 lookups
        # In Level 3 network state features, we aggregate Level 1 and Level 2 predictions
        st_congs_30 = list(lookup_st_30.values())
        tr_occ_30 = [p["occupancy_30"] for p in track_predictions]

        df_net_eng["pred_station_congestion_mean"] = np.mean(st_congs_30) if st_congs_30 else 0.0
        df_net_eng["pred_station_congestion_max"] = np.max(st_congs_30) if st_congs_30 else 0.0
        df_net_eng["pred_track_occupancy_mean"] = np.mean(tr_occ_30) if tr_occ_30 else 0.0
        df_net_eng["pred_track_occupancy_max"] = np.max(tr_occ_30) if tr_occ_30 else 0.0

        # -------------------------------------------------------------
        # LEVEL 3: Network Congestion Predictions
        # -------------------------------------------------------------
        network_predictions = self.network_predictor.predict_congestion(df_net_eng)

        # -------------------------------------------------------------
        # Compile Output FutureNetworkState Objects (15m, 30m, 60m horizons)
        # -------------------------------------------------------------
        tick = int(df_net.iloc[0]["tick"]) if not df_net.empty else 0
        state_id = str(df_net.iloc[0]["state_id"]) if not df_net.empty else "state_S00_T00"

        futures = {}
        for horizon in [15, 30, 60]:
            f_state = FutureNetworkState(horizon, tick, state_id)
            
            # Map station predictions
            for p in station_predictions:
                f_state.stations[p["station_id"]] = {
                    "congestion": p[f"congestion_{horizon}"],
                    "confidence": p["confidence"],
                    "prediction_interval": p["prediction_interval"]
                }
                
            # Map track predictions
            for p in track_predictions:
                f_state.tracks[p["track_id"]] = {
                    "occupancy": p[f"occupancy_{horizon}"],
                    "congestion": p["congestion_30"] if horizon == 30 else p[f"occupancy_{horizon}"],
                    "confidence": p["confidence"],
                    "prediction_interval": p["prediction_interval"]
                }

            # Map network predictions
            f_state.network = {
                "network_congestion": network_predictions[f"network_congestion_{horizon}"],
                "platform_utilization": network_predictions[f"platform_utilization_{horizon}"],
                "track_utilization": network_predictions[f"track_utilization_{horizon}"],
                "average_delay": network_predictions[f"average_delay_{horizon}"],
                "stress_index": network_predictions[f"stress_index_{horizon}"],
                "confidence": network_predictions["confidence"],
                "prediction_interval": network_predictions["prediction_interval"]
            }
            
            futures[horizon] = f_state.to_dict()

        return futures
