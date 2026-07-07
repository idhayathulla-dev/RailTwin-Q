class PropagationFeatureEngineer:
    @staticmethod
    def compile_train_propagation_features(train, network) -> dict:
        """
        Calculates propagation features for a specific Train node.
        """
        route = network.routes.get(train.route_id, [])
        stops_remaining = len(route) - (train.route_index + 1)
        
        # Recovery margin: baseline speed vs current speed
        recovery_margin = max(0.0, 110.0 - train.speed) # assuming 110.0 km/h baseline speed
        
        return {
            "train_id": train.train_no,
            "stops_remaining": stops_remaining,
            "priority": 1 if train.is_priority_train else 0,
            "current_delay": train.delay,
            "recovery_margin": recovery_margin,
            "speed_ratio": round(train.speed / 110.0, 2) if train.speed > 0 else 0.0
        }

    @staticmethod
    def compile_station_propagation_features(station, network) -> dict:
        """
        Calculates propagation features for a Station node.
        """
        incoming_delayed = 0
        for t in network.trains:
            if t.status == "MOVING" and t.delay > 5.0:
                # Check if train route path contains this station after current index
                route = network.routes.get(t.route_id, [])
                if station.station_id in route[t.route_index + 1:]:
                    incoming_delayed += 1

        platform_slack = max(0, station.platforms - station.platforms_occupied)
        
        return {
            "station_id": station.station_id,
            "incoming_delayed_trains": incoming_delayed,
            "platform_slack": platform_slack,
            "queue_pressure": station.trains_waiting,
            "propagation_pressure": incoming_delayed * 1.5 + station.trains_waiting * 2.0
        }

    @staticmethod
    def compile_track_propagation_features(track, network) -> dict:
        """
        Calculates propagation features for a Track node.
        """
        return {
            "track_id": track.track_id,
            "current_occupancy": track.occupancy_percent,
            "alternate_routes_count": 1 if track.track_id == 2 else 0, # alternate bypass tracks proxy
            "buffer_capacity": max(0, track.capacity - track.current_trains)
        }
