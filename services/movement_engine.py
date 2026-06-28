from models.railway_network import RailwayNetwork
from services.event_system import SignalFailureEvent, HeavyRainEvent, MaintenanceEvent, TrackBlockageEvent, PowerFailureEvent

def format_time(minutes):
    hrs = (8 + (int(minutes) // 60)) % 24
    mins = int(minutes) % 60
    return f"{hrs:02d}:{mins:02d}"

class MovementEngine:
    @staticmethod
    def tick(network: RailwayNetwork, active_events: list, current_time_minutes: int, dwell_time_default: int = 2):
        """
        Advances the simulation by 1 tick (1 minute).
        Updates train positions, progress, statuses, speeds, and delays.
        Also calculates distance statistics, expected arrival times, and delay reasons.
        """
        # Find active event modifiers
        rain_intensity = 0.0
        signal_failures = set()
        power_failures = set()
        maintenance_tracks = set()
        blocked_tracks = set()

        for event in active_events:
            if not event.active:
                continue
            if isinstance(event, HeavyRainEvent):
                rain_intensity = max(rain_intensity, event.intensity)
            elif isinstance(event, SignalFailureEvent):
                signal_failures.add(event.station_id)
            elif isinstance(event, PowerFailureEvent):
                power_failures.add(event.station_id)
            elif isinstance(event, MaintenanceEvent):
                maintenance_tracks.add(event.track_id)
            elif isinstance(event, TrackBlockageEvent):
                blocked_tracks.add(event.track_id)

        # Update track disruption flags
        for track in network.tracks:
            track.maintenance = track.track_id in maintenance_tracks
            track.blocked = track.track_id in blocked_tracks

        # Update each train
        for train in network.trains:
            route_stations = network.routes.get(train.route_id, [])
            if not route_stations:
                continue

            # Capture previous delay to calculate change
            prev_delay = train.delay
            
            # Reset temporary delay flags for this tick
            primary_reason = "NORMAL"
            secondary_reason = "NORMAL"

            # Check if train has arrived at the final station of the route
            if train.status == "ARRIVED" and train.current_station_id == route_stations[-1]:
                train.progress = 0.0
                train.current_track_id = None
                train.speed = 0.0
                train.delay_change_last_tick = 0.0
                train.primary_delay_reason = "NORMAL"
                train.secondary_delay_reason = "NORMAL"
                continue

            # Update train route_index if it's not set correctly
            if train.current_station_id in route_stations and train.route_index == 0:
                train.route_index = route_stations.index(train.current_station_id)

            # Pre-calculate schedule details if not set
            if not train.scheduled_arrival_time:
                # Calculate total route distance
                total_dist = 0.0
                for idx in range(len(route_stations) - 1):
                    s_curr = route_stations[idx]
                    s_next = route_stations[idx + 1]
                    for t in network.tracks:
                        if (t.source_station_id == s_curr and t.destination_station_id == s_next) or \
                           (t.source_station_id == s_next and t.destination_station_id == s_curr):
                            total_dist += t.distance
                            break
                
                # Scheduled travel time = (distance / speed) * 60 + dwell times
                num_dwells = max(0, len(route_stations) - 2)
                travel_time_mins = (total_dist / train.base_speed) * 60 + (num_dwells * dwell_time_default)
                train.scheduled_arrival_time = format_time(travel_time_mins)

            # Case 1: Train is at a station (status is WAITING, ARRIVED, or DELAYED with progress == 0.0)
            if train.status in ["WAITING", "ARRIVED", "DELAYED"] and train.progress == 0.0:
                train.speed = 0.0
                
                if train.dwell_time_remaining > 0:
                    # Dwell countdown
                    train.dwell_time_remaining -= 1
                    train.status = "ARRIVED"  # Stay at platform during dwell
                    
                    if train.dwell_time_remaining == 0:
                        train.status = "WAITING"
                else:
                    # Train wants to depart
                    # Check for signal or power failure at current station
                    if train.current_station_id in signal_failures or train.current_station_id in power_failures:
                        train.status = "DELAYED"
                        train.delay += 1
                        primary_reason = "SIGNAL_FAILURE"
                        continue

                    # Check if there is a next station on the route
                    if train.route_index + 1 < len(route_stations):
                        next_station_id = route_stations[train.route_index + 1]
                        
                        # Find track connecting current_station to next_station
                        target_track = None
                        for track in network.tracks:
                            if (track.source_station_id == train.current_station_id and track.destination_station_id == next_station_id) or \
                               (track.source_station_id == next_station_id and track.destination_station_id == train.current_station_id):
                                target_track = track
                                break

                        if target_track:
                            # Check if the track is blocked or under maintenance
                            if target_track.blocked:
                                train.status = "DELAYED"
                                train.delay += 1
                                primary_reason = "TRACK_BLOCKED"
                                continue
                            elif target_track.maintenance:
                                train.status = "DELAYED"
                                train.delay += 1
                                primary_reason = "MAINTENANCE"
                                continue
                            
                            # Check platform occupancy at next station: if destination is congested,
                            # optionally delay departure to avoid gridlock (WAITING_FOR_PLATFORM)
                            next_station = network.get_station_by_id(next_station_id)
                            if next_station and next_station.platforms_occupied >= next_station.platforms:
                                # Check if train is not high priority; let it wait
                                if not train.is_priority_train:
                                    train.status = "DELAYED"
                                    train.delay += 1
                                    primary_reason = "WAITING_FOR_PLATFORM"
                                    continue

                            # Depart to track!
                            train.status = "MOVING"
                            train.current_track_id = target_track.track_id
                            train.progress = 0.0
                            train.speed = train.base_speed
                        else:
                            train.status = "DELAYED"
                            train.delay += 1
                    else:
                        train.status = "ARRIVED"
                        train.progress = 0.0
                        train.current_track_id = None

            # Case 2: Train is moving along a track
            elif train.status == "MOVING":
                track = network.get_track_by_id(train.current_track_id)
                if not track:
                    train.status = "DELAYED"
                    train.delay += 1
                    continue

                # Check if the track segment is blocked or closed mid-transit
                if track.blocked:
                    train.speed = 0.0
                    train.delay += 1
                    primary_reason = "TRACK_BLOCKED"
                    continue
                elif track.maintenance:
                    train.speed = 0.0
                    train.delay += 1
                    primary_reason = "MAINTENANCE"
                    continue

                # Calculate speed reductions
                # 1. Rain reduction: e.g. up to 50% speed reduction
                rain_factor = 1.0 - (0.5 * rain_intensity)
                current_speed = train.base_speed * rain_factor
                current_speed = min(current_speed, track.max_speed)

                # Identify other trains on the same track ahead of this train
                other_trains_ahead = []
                for other in network.trains:
                    if other.train_no != train.train_no and other.status == "MOVING" and other.current_track_id == train.current_track_id:
                        if other.progress > train.progress:
                            other_trains_ahead.append(other)

                # 2. Block spacing slowing (Following train)
                following_slowdown = False
                if other_trains_ahead:
                    # Find the closest train ahead
                    closest_train = min(other_trains_ahead, key=lambda o: o.progress - train.progress)
                    progress_diff = closest_train.progress - train.progress
                    if progress_diff < 15.0: # Close proximity (within 15% track distance)
                        current_speed = min(current_speed, closest_train.speed * 0.7) # slow down behind it
                        following_slowdown = True

                # 3. Track congestion penalty
                if track.current_trains > 2:
                    congestion_factor = max(0.5, 1.0 - (0.15 * (track.current_trains - 1)))
                    current_speed *= congestion_factor

                train.speed = round(current_speed, 1)

                # Accumulate delay if speed is below scheduled speed
                if train.speed < train.base_speed:
                    speed_deficit_ratio = (train.base_speed - train.speed) / train.base_speed
                    train.delay += speed_deficit_ratio
                    
                    # Deduce primary delay reason
                    reasons = []
                    if rain_intensity > 0.0:
                        reasons.append("RAIN")
                    if following_slowdown:
                        reasons.append("FOLLOWING_TRAIN")
                    if track.current_trains > 2:
                        reasons.append("CONGESTION")
                    
                    if reasons:
                        primary_reason = reasons[0]
                        if len(reasons) > 1:
                            secondary_reason = reasons[1]

                # Advance progress
                distance_traveled = train.speed / 60.0
                progress_step = (distance_traveled / track.distance) * 100.0
                train.progress += progress_step

                # Round delay
                train.delay = round(train.delay, 2)

                # Check arrival at next station
                if train.progress >= 100.0:
                    train.progress = 0.0
                    train.route_index += 1
                    
                    if train.route_index < len(route_stations):
                        train.current_station_id = route_stations[train.route_index]
                        
                        if train.route_index == len(route_stations) - 1:
                            train.status = "ARRIVED"
                            train.current_track_id = None
                            train.speed = 0.0
                        else:
                            train.status = "ARRIVED" # Dwelling
                            train.dwell_time_remaining = dwell_time_default
                            train.current_track_id = None
                            train.speed = 0.0
                    else:
                        train.current_station_id = route_stations[-1]
                        train.status = "ARRIVED"
                        train.current_track_id = None
                        train.speed = 0.0

            # Calculate distance stats
            # 1. distance_travelled
            # Iterate previous stops in route
            distance_from_previous_edges = 0.0
            for idx in range(train.route_index):
                s_curr = route_stations[idx]
                s_next = route_stations[idx + 1]
                for t in network.tracks:
                    if (t.source_station_id == s_curr and t.destination_station_id == s_next) or \
                       (t.source_station_id == s_next and t.destination_station_id == s_curr):
                        distance_from_previous_edges += t.distance
                        break

            curr_track_len = 0.0
            if train.current_track_id:
                track = network.get_track_by_id(train.current_track_id)
                if track:
                    curr_track_len = track.distance

            current_edge_travelled = curr_track_len * (train.progress / 100.0)
            train.distance_travelled = round(distance_from_previous_edges + current_edge_travelled, 2)

            # 2. remaining_distance on current edge
            train.remaining_distance = round(curr_track_len * (1.0 - train.progress / 100.0), 2)

            # 3. remaining_route_distance
            # Iterate remaining edges in route
            distance_future_edges = 0.0
            # If train is moving, we start sum from route_index + 1
            # If train is at a station, we start from route_index
            start_index = train.route_index + 1 if train.current_track_id else train.route_index
            for idx in range(start_index, len(route_stations) - 1):
                s_curr = route_stations[idx]
                s_next = route_stations[idx + 1]
                for t in network.tracks:
                    if (t.source_station_id == s_curr and t.destination_station_id == s_next) or \
                       (t.source_station_id == s_next and t.destination_station_id == s_curr):
                        distance_future_edges += t.distance
                        break
            
            train.remaining_route_distance = round(train.remaining_distance + distance_future_edges, 2)

            # Calculate expected arrival time
            # Expected arrival = Scheduled departure time + optimal travel duration + delays
            # E.g. expected arrival is scheduled arrival + current delay
            # Let's parse scheduled arrival into minutes
            try:
                sch_hrs, sch_mins = map(int, train.scheduled_arrival_time.split(":"))
                sch_total_mins = sch_hrs * 60 + sch_mins
                exp_total_mins = sch_total_mins + int(train.delay)
                train.expected_arrival_time = format_time(exp_total_mins)
            except ValueError:
                train.expected_arrival_time = train.scheduled_arrival_time

            # Update delay change
            train.delay_change_last_tick = round(train.delay - prev_delay, 2)
            
            # Assign delay reasons
            if train.delay_change_last_tick > 0:
                train.primary_delay_reason = primary_reason
                train.secondary_delay_reason = secondary_reason
            else:
                train.primary_delay_reason = "NORMAL"
                train.secondary_delay_reason = "NORMAL"
