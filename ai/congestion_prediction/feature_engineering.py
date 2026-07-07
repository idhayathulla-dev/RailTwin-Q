import numpy as np
import pandas as pd
from ai.congestion_prediction.utils import train_logger

class FeatureEngineer:
    @staticmethod
    def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        time_col = "time" if "time" in df.columns else None
        
        if time_col and df[time_col].dtype == "object":
            try:
                df["hour"] = df[time_col].apply(lambda x: int(x.split(":")[0]) if isinstance(x, str) else 8)
                df["minute"] = df[time_col].apply(lambda x: int(x.split(":")[1]) if isinstance(x, str) else 0)
            except Exception:
                df["hour"] = 8
                df["minute"] = 0
        else:
            if "time_of_day" in df.columns:
                tod_mapping = {"MORNING": 8, "AFTERNOON": 14, "EVENING": 18, "NIGHT": 22}
                df["hour"] = df["time_of_day"].map(tod_mapping).fillna(8)
                df["minute"] = 0
            else:
                df["hour"] = 8
                df["minute"] = 0

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60.0)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60.0)
        return df

    @staticmethod
    def engineer_weather_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rain_col = "rain_intensity"
        if rain_col not in df.columns:
            if "weather_type" in df.columns:
                df[rain_col] = df["weather_type"].apply(lambda x: 0.8 if x == "RAINY" else 0.0)
            else:
                df[rain_col] = 0.0

        df["heavy_rain"] = df[rain_col].apply(lambda x: 1 if x >= 0.7 else 0)
        df["storm"] = df[rain_col].apply(lambda x: 1 if x >= 0.85 else 0)
        df["visibility"] = df[rain_col].apply(lambda x: round(10.0 * (1.0 - x * 0.9), 2))
        df["wind"] = df[rain_col].apply(lambda x: round(15.0 + x * 35.0, 2))
        df["temperature"] = df["hour"].apply(lambda h: round(22.0 + 8.0 * np.sin(2 * np.pi * (h - 8) / 24.0), 2))
        return df

    @staticmethod
    def engineer_station_features_v2(df_station: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Feature Engineering v2 additions onto the Station dataset.
        Computes ETA queues, decay scores, downstream backpressures, multi-hop aggregations,
        net flows, and route pressure features.
        """
        df_st = df_station.copy()
        df_tr = df_train.copy()

        # Sort values to ensure chronological order
        df_st = df_st.sort_values(by=["scenario_id", "station_id", "tick"]).reset_index(drop=True)
        df_tr = df_tr.sort_values(by=["scenario_id", "train_no", "tick"]).reset_index(drop=True)

        # -------------------------------------------------------------
        # Feature 1 & 2: ETA-Based Incoming Demand & Decay Weight Scores
        # -------------------------------------------------------------
        # Calculate ETA for each train record: ETA = remaining_distance / speed
        # If speed is 0 or train is waiting, assign high ETA or dwell remaining
        df_tr["eta_mins"] = np.where(
            df_tr["speed"] > 0,
            (df_tr["remaining_distance"] / df_tr["speed"]) * 60.0,
            999.0
        )
        # Cap ETA if train has already arrived
        df_tr["eta_mins"] = np.where(df_tr["status"] == "ARRIVED", 999.0, df_tr["eta_mins"])

        # Compile incoming demand stats per station
        eta_features = []
        for horizon in [5, 10, 15, 30, 60]:
            # Count trains heading to next_station with ETA within horizon
            df_tr[f"arr_in_{horizon}"] = np.where(df_tr["eta_mins"] <= horizon, 1, 0)
            
        df_tr["decay_score"] = np.exp(-df_tr["eta_mins"] / 15.0)

        # Aggregate train stats to station level (joining on next_station_id)
        # In train_dataset, next_station_id is represented as next_station (or similar). Let's check:
        # In data_loader, next_station was mapped to next_station_id. Let's support both column names.
        next_st_col = "next_station_id" if "next_station_id" in df_tr.columns else "next_station"
        
        train_agg = df_tr.groupby(["scenario_id", "tick", next_st_col]).agg(
            incoming_trains_eta_5=("arr_in_5", "sum"),
            incoming_trains_eta_10=("arr_in_10", "sum"),
            incoming_trains_eta_15=("arr_in_15", "sum"),
            incoming_trains_eta_30=("arr_in_30", "sum"),
            incoming_trains_eta_60=("arr_in_60", "sum"),
            incoming_demand_score=("decay_score", "sum")
        ).reset_index().rename(columns={next_st_col: "station_id"})

        # Merge demand scores into station dataset
        df_st = pd.merge(df_st, train_agg, on=["scenario_id", "tick", "station_id"], how="left")
        
        # Fill missing values (ticks with no incoming trains)
        fill_cols = [
            "incoming_trains_eta_5", "incoming_trains_eta_10", "incoming_trains_eta_15",
            "incoming_trains_eta_30", "incoming_trains_eta_60", "incoming_demand_score"
        ]
        df_st[fill_cols] = df_st[fill_cols].fillna(0.0)

        # -------------------------------------------------------------
        # Feature 3: Downstream Backpressure
        # -------------------------------------------------------------
        # In our linear railway network: 1 -> 2 -> 3 -> 4
        # Downstream for s is s+1. If s is 4, there is no downstream node.
        # We index the dataframe for fast downstream lookup
        st_lookup = df_st.set_index(["scenario_id", "tick", "station_id"])[
            ["station_congestion_score", "station_utilization_percent", "waiting_trains", "average_station_delay"]
        ].to_dict("index")

        def get_downstream_metrics(row):
            sc = row["scenario_id"]
            tick = row["tick"]
            s_id = row["station_id"]
            down_id = s_id + 1
            key = (sc, tick, down_id)
            if key in st_lookup:
                metrics = st_lookup[key]
                return (
                    metrics["station_congestion_score"],
                    metrics["station_utilization_percent"],
                    metrics["waiting_trains"],
                    metrics["average_station_delay"]
                )
            return (0.0, 0.0, 0.0, 0.0)

        downstream_vals = df_st.apply(get_downstream_metrics, axis=1)
        df_st["downstream_congestion"] = [v[0] for v in downstream_vals]
        df_st["downstream_platform_utilization"] = [v[1] for v in downstream_vals]
        df_st["downstream_queue_length"] = [v[2] for v in downstream_vals]
        df_st["downstream_average_delay"] = [v[3] for v in downstream_vals]

        # -------------------------------------------------------------
        # Feature 4: Multi-Hop Spatial Neighbor Aggregation
        # -------------------------------------------------------------
        # Chennai (1), Arakkonam (2), Katpadi (3), Jolarpettai (4)
        hops_map = {
            1: {1: [2], 2: [3], 3: [4]},
            2: {1: [1, 3], 2: [4], 3: []},
            3: {1: [2, 4], 2: [1], 3: []},
            4: {1: [3], 2: [2], 3: [1]}
        }

        # Lookup metrics by scenario, tick, station_id
        station_metrics_lookup = df_st.set_index(["scenario_id", "tick", "station_id"])[
            ["station_congestion_score", "average_station_delay", "station_utilization_percent", "waiting_trains"]
        ].to_dict("index")

        def get_hop_averages(row, hop: int):
            sc = row["scenario_id"]
            tick = row["tick"]
            s_id = row["station_id"]
            nodes = hops_map.get(s_id, {}).get(hop, [])
            
            congs, delays, utils, waits = [], [], [], []
            for n in nodes:
                key = (sc, tick, n)
                if key in station_metrics_lookup:
                    m = station_metrics_lookup[key]
                    congs.append(m["station_congestion_score"])
                    delays.append(m["average_station_delay"])
                    utils.append(m["station_utilization_percent"])
                    waits.append(m["waiting_trains"])
            
            return (
                np.mean(congs) if congs else 0.0,
                np.mean(delays) if delays else 0.0,
                np.mean(utils) if utils else 0.0,
                np.mean(waits) if waits else 0.0
            )

        for h in [1, 2, 3]:
            vals = df_st.apply(lambda r: get_hop_averages(r, h), axis=1)
            df_st[f"neighbor_congestion_mean_h{h}"] = [v[0] for v in vals]
            df_st[f"neighbor_delay_mean_h{h}"] = [v[1] for v in vals]
            df_st[f"neighbor_platform_utilization_h{h}"] = [v[2] for v in vals]
            df_st[f"neighbor_waiting_trains_h{h}"] = [v[3] for v in vals]

        # -------------------------------------------------------------
        # Feature 5: Net Flow & Rates
        # -------------------------------------------------------------
        df_st["net_flow"] = df_st["incoming_trains"] - df_st["outgoing_trains"]
        df_st["incoming_rate"] = df_st["incoming_trains"] / np.maximum(1.0, df_st["platforms_total"])
        df_st["outgoing_rate"] = df_st["outgoing_trains"] / np.maximum(1.0, df_st["platforms_total"])
        df_st["flow_balance_ratio"] = df_st["incoming_trains"] / np.maximum(1.0, df_st["outgoing_trains"])

        # -------------------------------------------------------------
        # Feature 7: Temporal Trend Features
        # -------------------------------------------------------------
        # Growth Rate: Value_T - Value_{T-1}
        trend_cols = ["station_congestion_score", "average_station_delay", "waiting_trains", "platforms_occupied"]
        for col in trend_cols:
            df_st[f"{col}_trend"] = df_st.groupby(["scenario_id", "station_id"])[col].diff().fillna(0.0)

        # -------------------------------------------------------------
        # Feature 8: Route Pressure Features
        # -------------------------------------------------------------
        # Chennai central is source for all routes. Jolarpettai is destination.
        # Chennai: 3 outgoing, 0 incoming, complexity 3.
        route_pressure_map = {
            1: {"incoming_routes": 0, "outgoing_routes": 3, "route_complexity": 3.0},
            2: {"incoming_routes": 3, "outgoing_routes": 3, "route_complexity": 2.0},
            3: {"incoming_routes": 3, "outgoing_routes": 3, "route_complexity": 2.0},
            4: {"incoming_routes": 3, "outgoing_routes": 0, "route_complexity": 1.0}
        }
        
        df_st["incoming_routes"] = df_st["station_id"].map(lambda x: route_pressure_map.get(x, {}).get("incoming_routes", 0))
        df_st["outgoing_routes"] = df_st["station_id"].map(lambda x: route_pressure_map.get(x, {}).get("outgoing_routes", 0))
        df_st["route_complexity"] = df_st["station_id"].map(lambda x: route_pressure_map.get(x, {}).get("route_complexity", 1.0))

        return df_st

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        df = FeatureEngineer.engineer_time_features(df)
        df = FeatureEngineer.engineer_weather_features(df)
        return df
