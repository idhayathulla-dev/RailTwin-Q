import os
import pandas as pd
import numpy as np
from ai.congestion_prediction.utils import train_logger
from ai.congestion_prediction.feature_engineering import FeatureEngineer

class DataLoader:
    @staticmethod
    def load_raw_datasets(data_dir: str = "datasets") -> tuple:
        """
        Loads the 5 raw CSV datasets.
        """
        df_train = pd.read_csv(os.path.join(data_dir, "train_dataset.csv"))
        df_station = pd.read_csv(os.path.join(data_dir, "station_dataset.csv"))
        df_track = pd.read_csv(os.path.join(data_dir, "track_dataset.csv"))
        df_network = pd.read_csv(os.path.join(data_dir, "network_state_dataset.csv"))
        df_prop = pd.read_csv(os.path.join(data_dir, "delay_propagation_dataset.csv"))
        return df_train, df_station, df_track, df_network, df_prop

    @staticmethod
    def prepare_station_dataset(df_station: pd.DataFrame, df_train: pd.DataFrame, df_network: pd.DataFrame) -> pd.DataFrame:
        """
        Builds Station dataset including rolling averages, centralities, neighbor aggregations,
        Feature Engineering v2 additions, and targets (+15, +30, +60).
        """
        train_logger.info("Preparing Station Congestion Dataset (Feature Engineering v2)...")
        df = df_station.copy()
        
        # 1. Feature Engineering v2: ETA demand, backpressure, multi-hop aggregates, route pressures
        df = FeatureEngineer.engineer_station_features_v2(df, df_train)

        # Sort chronologically to compute rolling averages
        df = df.sort_values(by=["scenario_id", "station_id", "tick"]).reset_index(drop=True)

        # 2. Feature 6 — Rolling Historical Features (Expanded)
        # Compute mean and max over windows
        rolling_cols_mean = [
            "platforms_occupied", "waiting_trains", "station_congestion_score", 
            "station_utilization_percent", "average_station_delay"
        ]
        for col in rolling_cols_mean:
            if col in df.columns:
                for w in [5, 10, 20]:
                    df[f"{col}_roll_mean_{w}"] = df.groupby(["scenario_id", "station_id"])[col].transform(
                        lambda x: x.rolling(w, min_periods=1).mean()
                    )

        # Rolling Max Congestion
        for w in [5, 10, 20]:
            df[f"station_congestion_score_roll_max_{w}"] = df.groupby(["scenario_id", "station_id"])["station_congestion_score"].transform(
                lambda x: x.rolling(w, min_periods=1).max()
            )

        # 3. Feature 9 — Network Pressure Features
        # Merge network statistics (platform/track utilization, global congestion, average network speed)
        net_sub = df_network[[
            "scenario_id", "tick", "network_congestion_score", "platform_utilization_percent",
            "track_utilization_percent", "average_network_delay", "average_train_speed", "network_resilience_score"
        ]].rename(columns={
            "network_congestion_score": "global_congestion_index",
            "platform_utilization_percent": "global_platform_utilization",
            "track_utilization_percent": "global_track_utilization",
            "average_network_delay": "global_average_delay",
            "average_train_speed": "global_average_speed",
            "network_resilience_score": "global_resilience_score"
        })
        df = pd.merge(df, net_sub, on=["scenario_id", "tick"], how="left")

        # 4. Merge Layer 2 predicted delays from train_dataset
        df_train_sub = df_train.groupby(["scenario_id", "tick", "current_station_id"])[
            ["future_delay_15", "future_delay_30", "future_delay_60"]
        ].mean().reset_index().rename(columns={"current_station_id": "station_id"})
        
        df = pd.merge(df, df_train_sub, on=["scenario_id", "tick", "station_id"], how="left")
        df.fillna({
            "future_delay_15": 0.0,
            "future_delay_30": 0.0,
            "future_delay_60": 0.0
        }, inplace=True)

        # 5. Generate targets (+15, +30, +60 minutes)
        for horizon in [15, 30, 60]:
            df[f"target_station_congestion_{horizon}"] = df.groupby(["scenario_id", "station_id"])["station_congestion_score"].shift(-horizon)

        # Drop rows where targets are NaN
        df.dropna(subset=[f"target_station_congestion_{h}" for h in [15, 30, 60]], inplace=True)
        return df

    @staticmethod
    def prepare_track_dataset(df_track: pd.DataFrame, df_train: pd.DataFrame, df_station_pred: pd.DataFrame = None) -> pd.DataFrame:
        """
        Builds Track dataset incorporating rolling averages, delay predictions, and Level 1 predicted station congestions.
        """
        train_logger.info("Preparing Track Congestion Dataset...")
        df = df_track.copy()
        df = df.sort_values(by=["scenario_id", "track_id", "tick"]).reset_index(drop=True)

        # 1. Rolling Historical Features
        rolling_cols = ["trains_on_track", "occupancy_percent", "average_speed"]
        for col in rolling_cols:
            for w in [5, 10, 20]:
                df[f"{col}_roll_{w}"] = df.groupby(["scenario_id", "track_id"])[col].transform(
                    lambda x: x.rolling(w, min_periods=1).mean()
                )

        # 2. Merge Delay Predictor targets
        df_train_sub = df_train.groupby(["scenario_id", "tick", "current_track_id"])[
            ["future_delay_15", "future_delay_30", "future_delay_60"]
        ].mean().reset_index().rename(columns={"current_track_id": "track_id"})
        df = pd.merge(df, df_train_sub, on=["scenario_id", "tick", "track_id"], how="left")
        df.fillna({
            "future_delay_15": 0.0,
            "future_delay_30": 0.0,
            "future_delay_60": 0.0
        }, inplace=True)

        # 3. Stacked Hierarchical Merging: Include Predicted Station Congestions (Level 1)
        if df_station_pred is None:
            df_station_lookup = pd.read_csv("datasets/station_dataset.csv")
            df_src = df_station_lookup[["scenario_id", "tick", "station_id", "station_congestion_score"]].rename(
                columns={"station_id": "source_station_id", "station_congestion_score": "src_station_congestion"}
            )
            df = pd.merge(df, df_src, on=["scenario_id", "tick", "source_station_id"], how="left")

            df_dest = df_station_lookup[["scenario_id", "tick", "station_id", "station_congestion_score"]].rename(
                columns={"station_id": "destination_station_id", "station_congestion_score": "dest_station_congestion"}
            )
            df = pd.merge(df, df_dest, on=["scenario_id", "tick", "destination_station_id"], how="left")
        else:
            df_src = df_station_pred[["scenario_id", "tick", "station_id", "pred_station_congestion_30"]].rename(
                columns={"station_id": "source_station_id", "pred_station_congestion_30": "src_station_congestion"}
            )
            df = pd.merge(df, df_src, on=["scenario_id", "tick", "source_station_id"], how="left")

            df_dest = df_station_pred[["scenario_id", "tick", "station_id", "pred_station_congestion_30"]].rename(
                columns={"station_id": "destination_station_id", "pred_station_congestion_30": "dest_station_congestion"}
            )
            df = pd.merge(df, df_dest, on=["scenario_id", "tick", "destination_station_id"], how="left")

        df.fillna({"src_station_congestion": 0.0, "dest_station_congestion": 0.0}, inplace=True)

        # 4. Generate targets (+15, +30, +60 minutes)
        for horizon in [15, 30, 60]:
            df[f"target_track_occupancy_{horizon}"] = df.groupby(["scenario_id", "track_id"])["occupancy_percent"].shift(-horizon)

        df.dropna(subset=[f"target_track_occupancy_{h}" for h in [15, 30, 60]], inplace=True)
        return df

    @staticmethod
    def prepare_network_dataset(df_network: pd.DataFrame, df_station_pred: pd.DataFrame = None, df_track_pred: pd.DataFrame = None) -> pd.DataFrame:
        """
        Builds Network dataset incorporating rolling averages, and Level 1 & Level 2 outputs.
        """
        train_logger.info("Preparing Network Congestion Dataset...")
        df = df_network.copy()
        df = df.sort_values(by=["scenario_id", "tick"]).reset_index(drop=True)

        # 1. Rolling Historical Features
        rolling_cols = ["network_congestion_score", "platform_utilization_percent", "track_utilization_percent", "average_network_delay"]
        for col in rolling_cols:
            for w in [5, 10, 20]:
                df[f"{col}_roll_{w}"] = df.groupby(["scenario_id"])[col].transform(
                    lambda x: x.rolling(w, min_periods=1).mean()
                )

        # 2. Stacked Hierarchical Merging: Include Predicted Station and Track Congestions
        if df_station_pred is None:
            df_station_lookup = pd.read_csv("datasets/station_dataset.csv")
            df_station_agg = df_station_lookup.groupby(["scenario_id", "tick"])["station_congestion_score"].agg(["mean", "max"]).reset_index().rename(
                columns={"mean": "pred_station_congestion_mean", "max": "pred_station_congestion_max"}
            )
        else:
            df_station_agg = df_station_pred.groupby(["scenario_id", "tick"])["pred_station_congestion_30"].agg(["mean", "max"]).reset_index().rename(
                columns={"mean": "pred_station_congestion_mean", "max": "pred_station_congestion_max"}
            )

        if df_track_pred is None:
            df_track_lookup = pd.read_csv("datasets/track_dataset.csv")
            df_track_agg = df_track_lookup.groupby(["scenario_id", "tick"])["occupancy_percent"].agg(["mean", "max"]).reset_index().rename(
                columns={"mean": "pred_track_occupancy_mean", "max": "pred_track_occupancy_max"}
            )
        else:
            df_track_agg = df_track_pred.groupby(["scenario_id", "tick"])["pred_track_occupancy_30"].agg(["mean", "max"]).reset_index().rename(
                columns={"mean": "pred_track_occupancy_mean", "max": "pred_track_occupancy_max"}
            )

        df = pd.merge(df, df_station_agg, on=["scenario_id", "tick"], how="left")
        df = pd.merge(df, df_track_agg, on=["scenario_id", "tick"], how="left")

        df.fillna({
            "pred_station_congestion_mean": 0.0, "pred_station_congestion_max": 0.0,
            "pred_track_occupancy_mean": 0.0, "pred_track_occupancy_max": 0.0
        }, inplace=True)

        # 3. Generate targets (+15, +30, +60 minutes)
        targets_dict = {
            "network_congestion_score": "target_network_congestion",
            "platform_utilization_percent": "target_platform_utilization",
            "track_utilization_percent": "target_track_utilization",
            "average_network_delay": "target_average_delay"
        }
        for source_col, target_prefix in targets_dict.items():
            for horizon in [15, 30, 60]:
                df[f"{target_prefix}_{horizon}"] = df.groupby(["scenario_id"])[source_col].shift(-horizon)

        all_target_cols = []
        for target_prefix in targets_dict.values():
            for h in [15, 30, 60]:
                all_target_cols.append(f"{target_prefix}_{h}")
        df.dropna(subset=all_target_cols, inplace=True)
        
        return df
