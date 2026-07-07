import numpy as np
import networkx as nx

class OperationalDependencyGraph:
    @staticmethod
    def clip(val, min_v=0.05, max_v=0.99):
        return round(float(np.clip(val, min_v, max_v)), 2)

    @staticmethod
    def add_resource_edges(G: nx.DiGraph, network):
        """
        Connects trains to the tracks they occupy, and tracks to their destination stations.
        Models how train progress delays affect track clearance and downstream arrivals.
        """
        for t in network.trains:
            if t.status == "ARRIVED":
                continue
                
            # If train is moving on a track
            if t.status == "MOVING" and t.current_track_id:
                track = next((tk for tk in network.tracks if tk.track_id == t.current_track_id), None)
                if track:
                    # Train delays track clearance
                    p = OperationalDependencyGraph.clip(
                        0.2 + 0.3 * (t.delay / 30.0) + 0.3 * (track.occupancy_percent / 100.0) + 0.1 * (1 if t.is_priority_train else 0)
                    )
                    G.add_edge(
                        f"train_{t.train_no}",
                        f"track_{t.current_track_id}",
                        propagation_probability=p,
                        expected_delay_transfer=round(t.delay * 0.8, 1),
                        transfer_time=10,
                        reason="Track Occupancy",
                        confidence=90.0
                    )
                    # Track delays dest station arrival
                    dest_st = track.destination_station_id
                    p_dest = OperationalDependencyGraph.clip(
                        0.1 + 0.4 * (track.occupancy_percent / 100.0) + 0.3 * (t.delay / 60.0)
                    )
                    G.add_edge(
                        f"track_{t.current_track_id}",
                        f"station_{dest_st}",
                        propagation_probability=p_dest,
                        expected_delay_transfer=round(t.delay * 0.6, 1),
                        transfer_time=15,
                        reason="Station Destination Queue",
                        confidence=85.0
                    )
            # If train is waiting at a station
            elif t.status in ["WAITING", "DWELLING"]:
                s = network.get_station_by_id(t.current_station_id)
                p_wait = OperationalDependencyGraph.clip(
                    0.1 + 0.5 * (s.station_congestion_score / 100.0) + 0.2 * (s.trains_waiting / 5.0)
                )
                G.add_edge(
                    f"station_{t.current_station_id}",
                    f"train_{t.train_no}",
                    propagation_probability=p_wait,
                    expected_delay_transfer=round(t.delay * 0.5, 1),
                    transfer_time=5,
                    reason="Platform Starvation Dwell",
                    confidence=80.0
                )

    @staticmethod
    def add_queue_edges(G: nx.DiGraph, network, active_events: list):
        """
        Models backpressure propagation. If station platforms are full,
        adjacent tracks saturate because incoming trains cannot enter.
        """
        weather_factor = 1.0
        for ev in active_events:
            if ev.__class__.__name__ == "HeavyRainEvent" and ev.active:
                weather_factor = 1.3

        for s in network.stations:
            # Check platform utilization
            if s.platforms_occupied >= s.platforms:
                p_back = OperationalDependencyGraph.clip(
                    0.3 + 0.5 * (s.station_congestion_score / 100.0) + 0.15 * weather_factor
                )
                # Platforms are full, block adjacent incoming tracks
                incoming_tracks = [tr for tr in network.tracks if tr.destination_station_id == s.station_id]
                for tr in incoming_tracks:
                    G.add_edge(
                        f"station_{s.station_id}",
                        f"track_{tr.track_id}",
                        propagation_probability=p_back,
                        expected_delay_transfer=20.0,
                        transfer_time=8,
                        reason="Junction Backpressure Hold",
                        confidence=95.0
                    )

    @staticmethod
    def add_shared_occupancy_edges(G: nx.DiGraph, network):
        """
        If multiple trains occupy the same track, the leading train's delay
        cascades directly onto trailing trains due to safety signaling blocks.
        """
        for tr in network.tracks:
            track_trains = [t for t in network.trains if t.current_track_id == tr.track_id and t.status == "MOVING"]
            if len(track_trains) < 2:
                continue
                
            track_trains = sorted(track_trains, key=lambda x: x.progress, reverse=True)
            
            for i in range(len(track_trains) - 1):
                lead = track_trains[i]
                trail = track_trains[i+1]
                
                p_block = OperationalDependencyGraph.clip(
                    0.4 + 0.4 * (lead.delay / 40.0) + 0.1 * (tr.occupancy_percent / 100.0)
                )
                # Transfer delay
                G.add_edge(
                    f"train_{lead.train_no}",
                    f"train_{trail.train_no}",
                    propagation_probability=p_block,
                    expected_delay_transfer=round(lead.delay * 0.9, 1),
                    transfer_time=5,
                    reason="Signal Block Safety Spacing",
                    confidence=92.0
                )
