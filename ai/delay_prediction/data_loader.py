import os
import pandas as pd
from ai.delay_prediction.utils import train_logger

class DataLoader:
    @staticmethod
    def load_and_merge_data(data_dir: str = "datasets") -> pd.DataFrame:
        """
        Loads the 5 railway datasets and merges them into a single consistent DataFrame.
        """
        train_logger.info("Starting data loading and integration...")

        # 1. Paths
        train_path = os.path.join(data_dir, "train_dataset.csv")
        station_path = os.path.join(data_dir, "station_dataset.csv")
        track_path = os.path.join(data_dir, "track_dataset.csv")
        network_path = os.path.join(data_dir, "network_state_dataset.csv")
        prop_path = os.path.join(data_dir, "delay_propagation_dataset.csv")

        # 2. Check existences
        for path in [train_path, station_path, track_path, network_path, prop_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required dataset file not found: {path}")

        # 3. Read CSVs
        df_train = pd.read_csv(train_path)
        df_station = pd.read_csv(station_path)
        df_track = pd.read_csv(track_path)
        df_network = pd.read_csv(network_path)
        df_prop = pd.read_csv(prop_path)

        train_logger.info(f"Loaded records count: Trains={len(df_train)}, Stations={len(df_station)}, Tracks={len(df_track)}, Network={len(df_network)}, Propagation={len(df_prop)}")

        # 4. Relational Joins
        # Join Network State: on scenario_id, tick, state_id
        # We prefix network features to avoid collision
        network_cols = [
            "scenario_id", "tick", "state_id", "network_congestion_score", 
            "average_network_delay", "platform_utilization_percent", 
            "track_utilization_percent", "active_disruptions", 
            "average_train_speed", "network_resilience_score"
        ]
        df_network_sub = df_network[network_cols]
        merged_df = pd.merge(
            df_train, df_network_sub, 
            on=["scenario_id", "tick", "state_id"], 
            how="left"
        )

        # Join Station: on scenario_id, tick, state_id, current_station_id == station_id
        station_cols = [
            "scenario_id", "tick", "state_id", "station_id", 
            "station_congestion_score", "station_utilization_percent", 
            "waiting_trains", "incoming_trains", "outgoing_trains", 
            "platforms_available", "average_station_delay",
            "node_degree", "betweenness_centrality", "closeness_centrality", 
            "station_connectivity"
        ]
        df_station_sub = df_station[station_cols].copy()
        # Rename station_utilization_percent to platform_utilization as requested by user
        df_station_sub.rename(columns={"station_utilization_percent": "platform_utilization"}, inplace=True)
        # Rename station_id to current_station_id to join
        df_station_sub.rename(columns={"station_id": "current_station_id"}, inplace=True)
        
        merged_df = pd.merge(
            merged_df, df_station_sub, 
            on=["scenario_id", "tick", "state_id", "current_station_id"], 
            how="left"
        )

        # Join Track: on scenario_id, tick, state_id, current_track_id == track_id
        track_cols = [
            "scenario_id", "tick", "state_id", "track_id", 
            "occupancy_percent", "track_capacity", "distance",
            "average_speed", "travel_time", "blocked"
        ]
        df_track_sub = df_track[track_cols].copy()
        # Rename columns to avoid collisions
        df_track_sub.rename(columns={
            "occupancy_percent": "track_occupancy",
            "average_speed": "track_avg_speed",
            "blocked": "track_blocked",
            "travel_time": "track_travel_time"
        }, inplace=True)
        df_track_sub.rename(columns={"track_id": "current_track_id"}, inplace=True)

        merged_df = pd.merge(
            merged_df, df_track_sub, 
            on=["scenario_id", "tick", "state_id", "current_track_id"], 
            how="left"
        )

        # Preprocess Propagation delay targets and details
        # Let's count how many trains each parent train affected at this tick
        parent_counts = df_prop.groupby(["scenario_id", "tick", "state_id", "parent_train"]).size().reset_index(name="number_of_affected_trains")
        
        # Merge parent_delay (look up parent train's delay at that tick in df_train)
        df_parent_delays = df_train[["scenario_id", "tick", "state_id", "train_no", "current_delay"]].rename(
            columns={"train_no": "parent_train", "current_delay": "parent_delay"}
        )
        
        df_prop_enriched = pd.merge(
            df_prop, parent_counts, 
            on=["scenario_id", "tick", "state_id", "parent_train"], 
            how="left"
        )
        df_prop_enriched = pd.merge(
            df_prop_enriched, df_parent_delays, 
            on=["scenario_id", "tick", "state_id", "parent_train"], 
            how="left"
        )

        # Join Propagation: on scenario_id, tick, state_id, train_no == child_train
        prop_cols = [
            "scenario_id", "tick", "state_id", "child_train",
            "propagation_depth", "delay_chain_length", "parent_delay",
            "number_of_affected_trains", "delay_transferred"
        ]
        df_prop_sub = df_prop_enriched[prop_cols].copy()
        df_prop_sub.rename(columns={
            "child_train": "train_no",
            "delay_transferred": "delay_transfer"
        }, inplace=True)

        merged_df = pd.merge(
            merged_df, df_prop_sub, 
            on=["scenario_id", "tick", "state_id", "train_no"], 
            how="left"
        )

        # Fill missing values from propagation (if a train has no delay propagation in that tick, set defaults)
        fill_vals = {
            "propagation_depth": 0,
            "delay_chain_length": 0,
            "parent_delay": 0.0,
            "number_of_affected_trains": 0,
            "delay_transfer": 0.0
        }
        merged_df.fillna(value=fill_vals, inplace=True)

        train_logger.info(f"Integrated dataset size: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        return merged_df
