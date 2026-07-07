import networkx as nx
from ai.delay_propagation.dependency_graph import OperationalDependencyGraph

class PropagationGraphBuilder:
    @staticmethod
    def build_propagation_graph(network, active_events: list, delay_preds: list, congestion_preds: dict) -> nx.DiGraph:
        """
        Dynamically constructs a Spatio-Temporal Operational Dependency Graph (DAG) for the current simulation tick.
        Nodes represent Trains, Stations, and Tracks.
        Edges represent spatial, temporal, and queue-based delay transfer paths.
        """
        G = nx.DiGraph()
        
        # 1. Add Train Nodes
        # Index delay predictions for fast lookup
        delay_map = {p["train_id"]: p["delay_predictions"] for p in delay_preds}
        delay_conf_map = {p["train_id"]: p["confidence"] for p in delay_preds}
        
        for t in network.trains:
            t_preds = delay_map.get(t.train_no, {"15": t.delay, "30": t.delay, "60": t.delay})
            t_conf = delay_conf_map.get(t.train_no, 90.0)
            G.add_node(
                f"train_{t.train_no}",
                type="train",
                train_no=t.train_no,
                name=t.name,
                current_delay=t.delay,
                pred_15=t_preds["15"],
                pred_30=t_preds["30"],
                pred_60=t_preds["60"],
                confidence=t_conf,
                status=t.status,
                route_id=t.route_id,
                route_index=t.route_index,
                priority=1 if t.is_priority_train else 0,
                speed=t.speed,
                remaining_distance=t.remaining_distance,
                remaining_stations=len(network.routes.get(t.route_id, [])) - (t.route_index + 1)
            )

        # 2. Add Station Nodes
        st_cong_30 = congestion_preds.get(30, {}).get("predicted_stations", {})
        for s in network.stations:
            pred_cong = st_cong_30.get(str(s.station_id), {}).get("congestion", s.station_congestion_score)
            G.add_node(
                f"station_{s.station_id}",
                type="station",
                station_id=s.station_id,
                name=s.name,
                platforms_total=s.platforms,
                platforms_occupied=s.platforms_occupied,
                trains_waiting=s.trains_waiting,
                current_congestion=s.station_congestion_score,
                pred_congestion_30=pred_cong,
                average_delay=s.average_station_delay
            )

        # 3. Add Track Nodes
        tr_occ_30 = congestion_preds.get(30, {}).get("predicted_tracks", {})
        for tr in network.tracks:
            pred_occ = tr_occ_30.get(tr.track_id, {}).get("occupancy", tr.occupancy_percent)
            G.add_node(
                f"track_{tr.track_id}",
                type="track",
                track_id=tr.track_id,
                capacity=tr.capacity,
                trains_on_track=tr.current_trains,
                current_occupancy=tr.occupancy_percent,
                pred_occupancy_30=pred_occ,
                blocked=tr.blocked
            )

        # 4. Construct Dependencies (Edges)
        OperationalDependencyGraph.add_resource_edges(G, network)
        OperationalDependencyGraph.add_queue_edges(G, network, active_events)
        OperationalDependencyGraph.add_shared_occupancy_edges(G, network)
        
        return G
