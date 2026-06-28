from models.railway_network import RailwayNetwork

class StateEngine:
    @staticmethod
    def update_occupancies(network: RailwayNetwork):
        """
        Updates the occupancy metrics of stations and tracks based on
        current train locations, status, and progress.
        Also calculates average speeds, travel times, utilizations, congestion,
        and incoming/outgoing train counts.
        """
        # Reset all station variables
        for station in network.stations:
            station.trains_waiting = 0
            station.platforms_occupied = 0
            station.incoming_trains = 0
            station.outgoing_trains = 0
            station.average_station_delay = 0.0
            station.station_utilization_percent = 0.0
            station.station_congestion_score = 0.0

        # Reset all track variables
        for track in network.tracks:
            track.current_trains = 0
            track.occupancy_percent = 0.0
            track.average_speed = float(track.max_speed)
            track.travel_time = (track.distance / track.max_speed) * 60.0 # in minutes
            
        # Dictionary to track train speeds on each track to calculate average
        track_train_speeds = {t.track_id: [] for t in network.tracks}
        station_train_delays = {s.station_id: [] for s in network.stations}

        # Process each train
        for train in network.trains:
            route_stations = network.routes.get(train.route_id, [])
            if not route_stations:
                continue
                
            # Check where the train is
            if train.progress == 0.0 or train.status in ["WAITING", "ARRIVED", "DELAYED"]:
                # Train is at a station
                station = network.get_station_by_id(train.current_station_id)
                if station:
                    station_train_delays[station.station_id].append(train.delay)
                    if train.status in ["WAITING", "DELAYED"] and train.dwell_time_remaining == 0:
                        station.trains_waiting += 1
                    else:
                        station.platforms_occupied = min(station.platforms_occupied + 1, station.platforms)
            
            elif train.status == "MOVING" and train.current_track_id is not None:
                # Train is on a track
                track = network.get_track_by_id(train.current_track_id)
                if track:
                    track.current_trains += 1
                    track_train_speeds[track.track_id].append(train.speed)

                    # Update incoming and outgoing train counters for station
                    # A train moves from current_station towards next_station in route
                    next_station_id = None
                    if train.route_index + 1 < len(route_stations):
                        next_station_id = route_stations[train.route_index + 1]
                    
                    if next_station_id is not None:
                        # Destination is incoming
                        dest_station = network.get_station_by_id(next_station_id)
                        if dest_station:
                            dest_station.incoming_trains += 1
                        
                        # Source is outgoing
                        src_station = network.get_station_by_id(train.current_station_id)
                        if src_station:
                            src_station.outgoing_trains += 1

        # Post-process station metrics
        for station in network.stations:
            # Utilization
            if station.platforms > 0:
                station.station_utilization_percent = round((station.platforms_occupied / station.platforms) * 100.0, 2)
                # Congestion score
                station.station_congestion_score = round(((station.platforms_occupied + station.trains_waiting) / station.platforms) * 100.0, 2)
            
            # Average station delay
            delays = station_train_delays.get(station.station_id, [])
            if delays:
                station.average_station_delay = round(sum(delays) / len(delays), 2)

        # Post-process track metrics
        for track in network.tracks:
            # Occupancy percent
            if track.capacity > 0:
                track.occupancy_percent = round((track.current_trains / track.capacity) * 100.0, 2)
            
            # Average speed and travel time
            speeds = track_train_speeds.get(track.track_id, [])
            if speeds:
                track.average_speed = round(sum(speeds) / len(speeds), 1)
            else:
                track.average_speed = float(track.max_speed)
                
            if track.average_speed > 0:
                track.travel_time = round((track.distance / track.average_speed) * 60.0, 2)
            else:
                track.travel_time = float('inf')

    history = []

    @classmethod
    def clear_history(cls):
        cls.history = []

    @classmethod
    def record_snapshot(cls, network: RailwayNetwork, sim_time: str, active_events: list):
        """
        Takes a snapshot of the network state and appends it to the history list.
        """
        snapshot = cls.get_state_snapshot(network)
        snapshot["time"] = sim_time
        
        # Include current active event names with their properties
        serialized_events = []
        for e in active_events:
            if e.active:
                event_dict = {"name": e.name, "duration": e.duration}
                if hasattr(e, "intensity"):
                    event_dict["intensity"] = e.intensity
                if hasattr(e, "station_id"):
                    event_dict["station_id"] = e.station_id
                if hasattr(e, "track_id"):
                    event_dict["track_id"] = e.track_id
                serialized_events.append(event_dict)
                
        snapshot["active_events"] = serialized_events
        cls.history.append(snapshot)
        return snapshot

    @staticmethod
    def get_state_snapshot(network: RailwayNetwork) -> dict:
        """
        Generates a summary state snapshot dictionary of the entire network.
        Includes all attributes for trains, stations, and tracks.
        """
        active_trains = len(network.trains)
        congested_stations = sum(1 for s in network.stations if s.platforms_occupied >= s.platforms * 0.8)
        congested_tracks = sum(1 for t in network.tracks if t.occupancy_percent >= 80.0)

        # Calculate average train speed and average delay
        avg_speed = 0.0
        avg_delay = 0.0
        if active_trains > 0:
            avg_speed = sum(t.speed for t in network.trains) / active_trains
            avg_delay = sum(t.delay for t in network.trains) / active_trains

        return {
            "active_trains": active_trains,
            "congested_stations": congested_stations,
            "congested_tracks": congested_tracks,
            "average_speed": round(avg_speed, 2),
            "average_delay": round(avg_delay, 2),
            "trains": [
                {
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
                } for t in network.trains
            ],
            "stations": [
                {
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
                    "node_degree": s.node_degree,
                    "betweenness_centrality": s.betweenness_centrality,
                    "closeness_centrality": s.closeness_centrality,
                    "station_connectivity": s.station_connectivity
                } for s in network.stations
            ],
            "tracks": [
                {
                    "track_id": tr.track_id,
                    "source": tr.source_station_id,
                    "destination": tr.destination_station_id,
                    "track_capacity": tr.capacity,
                    "distance": tr.distance,
                    "max_speed": tr.max_speed,
                    "trains_on_track": tr.current_trains,
                    "occupancy_percent": round(tr.occupancy_percent, 2),
                    "average_speed": tr.average_speed,
                    "travel_time": tr.travel_time,
                    "blocked": tr.blocked,
                    "maintenance": tr.maintenance
                } for tr in network.tracks
            ]
        }
