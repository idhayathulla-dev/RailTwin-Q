import networkx as nx

class TrainDatasetGenerator:
    @staticmethod
    def generate(scenario_id: int, history_records: list, network_routes: dict, stations_map: dict, tracks_map: dict, graph: nx.Graph) -> list:
        rows = []
        num_ticks = len(history_records)

        # 1. Precalculate route lengths and station degree/centrality lists to avoid duplicate calculations
        # Get average centrality metrics across all nodes
        betweenness = nx.betweenness_centrality(graph, weight="weight")
        avg_betweenness = sum(betweenness.values()) / len(betweenness) if betweenness else 0.0

        route_lengths = {}
        route_junction_counts = {}
        route_critical_counts = {}

        for route_id, route_stations in network_routes.items():
            # Route length
            r_len = 0.0
            for idx in range(len(route_stations) - 1):
                u, v = route_stations[idx], route_stations[idx + 1]
                for track in tracks_map.values():
                    if (track.source_station_id == u and track.destination_station_id == v) or \
                       (track.source_station_id == v and track.destination_station_id == u):
                        r_len += track.distance
                        break
            route_lengths[route_id] = round(r_len, 2)

            # Junctions (nodes with degree > 2)
            j_count = sum(1 for s in route_stations if graph.degree(s) > 2)
            route_junction_counts[route_id] = j_count

            # Critical stations (centrality > average betweenness)
            crit_count = sum(1 for s in route_stations if betweenness.get(s, 0.0) > avg_betweenness)
            route_critical_counts[route_id] = crit_count

        for i, (tick, snap) in enumerate(history_records):
            time_str = snap["time"]
            state_id = f"state_S{scenario_id:03d}_T{tick:03d}"
            
            # Events
            active_events = snap.get("active_events", [])
            rain_intensity = 0.0
            is_festival_rush = False
            signal_fail_stations = set()
            maintenance_tracks = set()
            blocked_tracks = set()
            
            for ev in active_events:
                name = ev["name"]
                if name == "Heavy Rain":
                    rain_intensity = ev.get("intensity", 0.0)
                elif name == "Signal Failure":
                    signal_fail_stations.add(ev.get("station_id"))
                elif name == "Maintenance":
                    maintenance_tracks.add(ev.get("track_id"))
                elif name == "Track Blockage":
                    blocked_tracks.add(ev.get("track_id"))
                elif name == "Festival Rush":
                    is_festival_rush = True
            
            weather_type = "RAINY" if rain_intensity > 0.0 else "CLEAR"
            
            for train in snap["trains"]:
                train_no = train["train_no"]
                curr_station_id = train["current_station"]
                route_id = train["route_id"]
                status = train["status"]
                progress = train["progress"]
                speed = train["speed"]
                max_speed = train["max_speed"]
                current_delay = train["delay"]
                delay_change = train["delay_change_last_tick"]
                dist_travelled = train["distance_travelled"]
                rem_dist = train["remaining_distance"]
                rem_route_dist = train["remaining_route_distance"]
                sch_arr = train["scheduled_arrival_time"]
                exp_arr = train["expected_arrival_time"]
                primary_reason = train["primary_delay_reason"]
                secondary_reason = train["secondary_delay_reason"]
                curr_track_id = train["current_track_id"]
                
                # Fetch route stops
                route_stations = network_routes.get(route_id, [])
                next_station_id = None
                curr_idx = 0
                
                if curr_station_id in route_stations:
                    curr_idx = route_stations.index(curr_station_id)
                    if curr_idx + 1 < len(route_stations):
                        next_station_id = route_stations[curr_idx + 1]
                
                curr_station_name = stations_map.get(curr_station_id, f"Station {curr_station_id}")
                next_station_name = stations_map.get(next_station_id, "N/A") if next_station_id else "N/A"
                
                # Dynamic Route Complexity Metrics
                remaining_stops = max(0, len(route_stations) - curr_idx - 1)
                route_length = route_lengths.get(route_id, 0.0)
                junction_count = route_junction_counts.get(route_id, 0)
                critical_station_count = route_critical_counts.get(route_id, 0)

                # Alternative paths (using simple NetworkX paths to destination station)
                alternative_routes = 1
                if next_station_id and route_stations:
                    try:
                        dest_id = route_stations[-1]
                        alternative_routes = len(list(nx.all_simple_paths(graph, curr_station_id, dest_id)))
                    except Exception:
                        alternative_routes = 1

                # Disruption flags at train level
                is_signal_failure = curr_station_id in signal_fail_stations
                is_maintenance = curr_track_id in maintenance_tracks if curr_track_id else False
                is_blockage = curr_track_id in blocked_tracks if curr_track_id else False
                
                # Track features
                track_occupancy = 0.0
                track_speed_limit = 0.0
                track_capacity = 0
                trains_on_track = 0
                
                if curr_track_id:
                    for tr in snap["tracks"]:
                        if tr["track_id"] == curr_track_id:
                            track_occupancy = tr["occupancy_percent"]
                            track_speed_limit = tr["max_speed"]
                            track_capacity = tr["track_capacity"]
                            trains_on_track = tr["trains_on_track"]
                            break
                            
                # Station occupancy
                station_occupancy = 0.0
                for st in snap["stations"]:
                    if st["station_id"] == curr_station_id:
                        station_occupancy = st["station_utilization_percent"]
                        break
                        
                # Lookahead targets
                future_delay_15 = current_delay
                future_idx_15 = min(i + 15, num_ticks - 1)
                for f_train in history_records[future_idx_15][1]["trains"]:
                    if f_train["train_no"] == train_no:
                        future_delay_15 = f_train["delay"]
                        break
                        
                future_delay_30 = current_delay
                future_idx_30 = min(i + 30, num_ticks - 1)
                for f_train in history_records[future_idx_30][1]["trains"]:
                    if f_train["train_no"] == train_no:
                        future_delay_30 = f_train["delay"]
                        break
                        
                future_delay_60 = current_delay
                future_idx_60 = min(i + 60, num_ticks - 1)
                for f_train in history_records[future_idx_60][1]["trains"]:
                    if f_train["train_no"] == train_no:
                        future_delay_60 = f_train["delay"]
                        break
                        
                rows.append({
                    "scenario_id": scenario_id,
                    "tick": tick,
                    "state_id": state_id,
                    "simulation_time_minutes": tick,
                    "time": time_str,
                    "train_no": train_no,
                    "train_name": train["name"],
                    "train_type": train["train_type"],
                    "is_priority_train": 1 if train["is_priority_train"] else 0,
                    "route_id": route_id,
                    "current_station_id": curr_station_id,
                    "current_station": curr_station_name,
                    "next_station_id": next_station_id if next_station_id else -1,
                    "next_station": next_station_name,
                    "status": status,
                    "progress": progress,
                    "speed": speed,
                    "max_speed": max_speed,
                    "current_delay": current_delay,
                    "delay_change_last_tick": delay_change,
                    "distance_travelled": dist_travelled,
                    "remaining_distance": rem_dist,
                    "remaining_route_distance": rem_route_dist,
                    "scheduled_arrival_time": sch_arr,
                    "expected_arrival_time": exp_arr,
                    "weather_type": weather_type,
                    "rain_intensity": round(rain_intensity, 2),
                    "signal_failure": 1 if is_signal_failure else 0,
                    "maintenance": 1 if is_maintenance else 0,
                    "track_blockage": 1 if is_blockage else 0,
                    "festival_rush": 1 if is_festival_rush else 0,
                    "station_occupancy": station_occupancy,
                    "track_occupancy": track_occupancy,
                    "track_speed_limit": track_speed_limit,
                    "track_capacity": track_capacity,
                    "trains_on_track": trains_on_track,
                    "remaining_stations": remaining_stops,
                    "route_length": route_length,
                    "junction_count": junction_count,
                    "alternative_routes": alternative_routes,
                    "critical_station_count": critical_station_count,
                    "future_delay_15": round(future_delay_15, 2),
                    "future_delay_30": round(future_delay_30, 2),
                    "future_delay_60": round(future_delay_60, 2),
                    "primary_delay_reason": primary_reason,
                    "secondary_delay_reason": secondary_reason,
                    "current_track_id": curr_track_id if curr_track_id else -1
                })

        return rows
