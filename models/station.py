class Station:
    def __init__(
        self,
        station_id,
        name,
        latitude,
        longitude,
        platforms
    ):
        self.station_id = station_id
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.platforms = platforms

        # State / occupancy variables
        self.trains_waiting = 0
        self.platforms_occupied = 0
        self.incoming_trains = 0
        self.outgoing_trains = 0
        
        # Performance metrics
        self.station_congestion_score = 0.0
        self.station_utilization_percent = 0.0
        self.average_station_delay = 0.0
        
        # Graph metrics (NetworkX)
        self.node_degree = 0
        self.betweenness_centrality = 0.0
        self.closeness_centrality = 0.0
        self.station_connectivity = 0.0

        # Targets
        self.future_station_congestion = 0.0

    def __str__(self):
        return f"{self.name}"
