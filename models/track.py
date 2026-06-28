class Track:
    def __init__(
        self,
        track_id,
        source_station_id,
        destination_station_id,
        distance,
        max_speed,
        capacity=5
    ):
        self.track_id = track_id
        self.source_station_id = source_station_id
        self.destination_station_id = destination_station_id
        self.distance = distance
        self.max_speed = max_speed
        self.capacity = capacity

        # State / occupancy track variables
        self.current_trains = 0
        self.occupancy_percent = 0.0
        self.average_speed = 0.0
        self.travel_time = 0.0 # expected travel time in minutes under current conditions
        
        # Disruption flags
        self.blocked = False
        self.maintenance = False

        # Targets
        self.future_track_congestion = 0.0

    def __str__(self):
        return f"Track {self.track_id}: Station {self.source_station_id} -> Station {self.destination_station_id}"
