class Train:
    def __init__(
        self,
        train_no,
        name,
        route_id,
        current_station_id,
        speed,
        delay,
        train_type="PASSENGER",
        max_speed=110
    ):
        self.train_no = train_no
        self.name = name
        self.route_id = route_id
        self.current_station_id = current_station_id
        
        # Speed attributes
        self.base_speed = speed
        self.speed = speed
        self.max_speed = max_speed
        self.delay = delay
        self.delay_change_last_tick = 0.0

        # Train categorization
        self.train_type = train_type # PASSENGER, EXPRESS, SUPERFAST, FREIGHT
        self.is_priority_train = train_type in ["EXPRESS", "SUPERFAST"]

        # Movement tracking
        self.status = "WAITING"  # Starts at the station waiting to depart
        self.progress = 0.0      # Progress along the track (0.0 to 100.0)
        self.route_index = 0     # Index in route stations list
        self.dwell_time_remaining = 0
        self.current_track_id = None

        # Distance tracking
        self.distance_travelled = 0.0
        self.remaining_distance = 0.0
        self.remaining_route_distance = 0.0

        # Arrival schedules
        self.scheduled_arrival_time = ""
        self.expected_arrival_time = ""

        # Delay tracking
        self.primary_delay_reason = "NORMAL"
        self.secondary_delay_reason = "NORMAL"

    def __str__(self):
        return f"{self.train_no} - {self.name} ({self.train_type})"
