import pandas as pd
import networkx as nx
from ai.delay_prediction.utils import pred_logger
from ai.delay_prediction.inference import InferenceEngine
from services.synthetic_data.feature_engineering import FeatureEngineering
from services.synthetic_data.train_dataset_generator import TrainDatasetGenerator
from services.synthetic_data.station_dataset_generator import StationDatasetGenerator
from services.synthetic_data.track_dataset_generator import TrackDatasetGenerator
from services.synthetic_data.network_dataset_generator import NetworkDatasetGenerator
from services.synthetic_data.propagation_dataset_generator import PropagationDatasetGenerator
from services.graph_builder import GraphBuilder

class PredictionService:
    def __init__(self, models_dir="models/delay_predictor"):
        self.inference_engine = InferenceEngine(models_dir)
        self.inference_engine.initialize()

    def get_predictions_for_tick(self, network, tick: int, time_str: str, running_events: list) -> list:
        """
        Extracts the current tick's state, builds the unified features dataframe, and runs delay predictions.
        """
        if not self.inference_engine.initialized:
            pred_logger.warning("Inference engine not initialized. Cannot predict.")
            return []

        # 1. Build a temporary snapshot for the current tick
        # This matches the StateEngine format exactly
        active_events_serialized = []
        for ev in running_events:
            ev_data = {
                "name": ev.__class__.__name__.replace("Event", ""),
                "active": ev.active,
                "duration": ev.duration
            }
            # Add specific fields
            if hasattr(ev, "intensity"):
                ev_data["intensity"] = ev.intensity
            if hasattr(ev, "station_id"):
                ev_data["station_id"] = ev.station_id
            if hasattr(ev, "track_id"):
                ev_data["track_id"] = ev.track_id
            active_events_serialized.append(ev_data)

        # Get centralities dynamically
        graph = GraphBuilder.build_graph(network)
        graph_metrics = FeatureEngineering.compute_graph_metrics(graph)

        trains_serialized = []
        for t in network.trains:
            trains_serialized.append({
                "train_no": t.train_no,
                "name": t.name,
                "train_type": t.train_type,
                "is_priority_train": t.is_priority_train,
                "route_id": t.route_id,
                "current_station": t.current_station_id,
                "status": t.status,
                "progress": round(t.progress, 2),
                "speed": t.speed,
                "max_speed": t.max_speed,
                "delay": t.delay,
                "delay_change_last_tick": t.delay_change_last_tick,
                "distance_travelled": t.distance_travelled,
                "remaining_distance": t.remaining_distance,
                "remaining_route_distance": t.remaining_route_distance,
                "scheduled_arrival_time": t.scheduled_arrival_time,
                "expected_arrival_time": t.expected_arrival_time,
                "primary_delay_reason": t.primary_delay_reason,
                "secondary_delay_reason": t.secondary_delay_reason,
                "current_track_id": t.current_track_id
            })

        stations_serialized = []
        for s in network.stations:
            metrics = graph_metrics.get(s.station_id, {})
            stations_serialized.append({
                "station_id": s.station_id,
                "name": s.name,
                "platforms_total": s.platforms,
                "platforms_occupied": s.platforms_occupied,
                "trains_waiting": s.trains_waiting,
                "incoming_trains": s.incoming_trains,
                "outgoing_trains": s.outgoing_trains,
                "station_congestion_score": s.station_congestion_score,
                "station_utilization_percent": s.station_utilization_percent,
                "average_station_delay": s.average_station_delay,
                "node_degree": metrics.get("node_degree", 0),
                "betweenness_centrality": metrics.get("betweenness_centrality", 0.0),
                "closeness_centrality": metrics.get("closeness_centrality", 0.0),
                "station_connectivity": metrics.get("station_connectivity", 0.0)
            })

        tracks_serialized = []
        for tr in network.tracks:
            tracks_serialized.append({
                "track_id": tr.track_id,
                "source": tr.source_station_id,
                "destination": tr.destination_station_id,
                "distance": tr.distance,
                "track_capacity": tr.capacity,
                "max_speed": tr.max_speed,
                "trains_on_track": tr.current_trains,
                "occupancy_percent": tr.occupancy_percent,
                "average_speed": tr.average_speed,
                "travel_time": tr.travel_time,
                "blocked": tr.blocked,
                "maintenance": tr.maintenance
            })

        congested_st = sum(1 for s in network.stations if s.station_utilization_percent >= 80.0)
        congested_tr = sum(1 for tr in network.tracks if tr.occupancy_percent >= 80.0)
        tot_active_trains = sum(1 for t in network.trains if t.status != "ARRIVED")
        avg_delay = sum(t.delay for t in network.trains) / len(network.trains) if network.trains else 0.0
        avg_speed = sum(t.speed for t in network.trains) / len(network.trains) if network.trains else 0.0

        snapshot = {
            "time": time_str,
            "trains": trains_serialized,
            "stations": stations_serialized,
            "tracks": tracks_serialized,
            "active_events": active_events_serialized,
            "congested_stations": congested_st,
            "congested_tracks": congested_tr,
            "active_trains": tot_active_trains,
            "average_delay": round(avg_delay, 2),
            "average_speed": round(avg_speed, 2)
        }

        # 2. Build dataset generators on a single-tick history record
        history_records = [(tick, snapshot)]
        stations_map = {s.station_id: s.name for s in network.stations}
        tracks_map = {t.track_id: t for t in network.tracks}

        trains_data = TrainDatasetGenerator.generate(1, history_records, network.routes, stations_map, tracks_map, graph)
        stations_data = StationDatasetGenerator.generate(1, history_records, stations_map)
        tracks_data = TrackDatasetGenerator.generate(1, history_records, stations_map)
        network_data = NetworkDatasetGenerator.generate(1, history_records, "WEEKDAY")
        prop_data = PropagationDatasetGenerator.generate(1, history_records, stations_map)

        if not trains_data:
            return []

        # 3. Load into DataFrames
        df_train = pd.DataFrame(trains_data)
        df_station = pd.DataFrame(stations_data)
        df_track = pd.DataFrame(tracks_data)
        df_network = pd.DataFrame(network_data)
        df_prop = pd.DataFrame(prop_data)

        # 4. Merge slices (equivalent to data_loader.py)
        # Prefix network
        network_cols = [
            "scenario_id", "tick", "state_id", "network_congestion_score", 
            "average_network_delay", "platform_utilization_percent", 
            "track_utilization_percent", "active_disruptions", 
            "average_train_speed", "network_resilience_score"
        ]
        df_network_sub = df_network[network_cols]
        merged_df = pd.merge(df_train, df_network_sub, on=["scenario_id", "tick", "state_id"], how="left")

        # Join Station
        station_cols = [
            "scenario_id", "tick", "state_id", "station_id", 
            "station_congestion_score", "station_utilization_percent", 
            "waiting_trains", "incoming_trains", "outgoing_trains", 
            "platforms_available", "average_station_delay",
            "node_degree", "betweenness_centrality", "closeness_centrality", 
            "station_connectivity"
        ]
        df_station_sub = df_station[station_cols].copy()
        df_station_sub.rename(columns={"station_utilization_percent": "platform_utilization", "station_id": "current_station_id"}, inplace=True)
        merged_df = pd.merge(merged_df, df_station_sub, on=["scenario_id", "tick", "state_id", "current_station_id"], how="left")

        # Join Track
        track_cols = [
            "scenario_id", "tick", "state_id", "track_id", 
            "occupancy_percent", "track_capacity", "distance",
            "average_speed", "travel_time", "blocked"
        ]
        df_track_sub = df_track[track_cols].copy()
        df_track_sub.rename(columns={
            "occupancy_percent": "track_occupancy",
            "average_speed": "track_avg_speed",
            "blocked": "track_blocked",
            "travel_time": "track_travel_time",
            "track_id": "current_track_id"
        }, inplace=True)
        merged_df = pd.merge(merged_df, df_track_sub, on=["scenario_id", "tick", "state_id", "current_track_id"], how="left")

        # Join Propagation (if any)
        if not df_prop.empty:
            parent_counts = df_prop.groupby(["scenario_id", "tick", "state_id", "parent_train"]).size().reset_index(name="number_of_affected_trains")
            df_parent_delays = df_train[["scenario_id", "tick", "state_id", "train_no", "current_delay"]].rename(
                columns={"train_no": "parent_train", "current_delay": "parent_delay"}
            )
            df_prop_enriched = pd.merge(df_prop, parent_counts, on=["scenario_id", "tick", "state_id", "parent_train"], how="left")
            df_prop_enriched = pd.merge(df_prop_enriched, df_parent_delays, on=["scenario_id", "tick", "state_id", "parent_train"], how="left")

            prop_cols = [
                "scenario_id", "tick", "state_id", "child_train",
                "propagation_depth", "delay_chain_length", "parent_delay",
                "number_of_affected_trains", "delay_transferred"
            ]
            df_prop_sub = df_prop_enriched[prop_cols].copy()
            df_prop_sub.rename(columns={"child_train": "train_no", "delay_transferred": "delay_transfer"}, inplace=True)
            merged_df = pd.merge(merged_df, df_prop_sub, on=["scenario_id", "tick", "state_id", "train_no"], how="left")

        # Fill missing propagation fields
        fill_vals = {
            "propagation_depth": 0,
            "delay_chain_length": 0,
            "parent_delay": 0.0,
            "number_of_affected_trains": 0,
            "delay_transfer": 0.0
        }
        for k, v in fill_vals.items():
            if k not in merged_df.columns:
                merged_df[k] = v
            else:
                merged_df[k] = merged_df[k].fillna(v)

        # 5. Predict via InferenceEngine
        predictions = self.inference_engine.predict_train_delays(merged_df)
        return predictions
