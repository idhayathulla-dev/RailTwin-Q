class RailwayNetwork:
    def __init__(self):
        self.stations = []
        self.tracks = []
        self.trains = []
        self.routes = {}

    def add_station(self, station):
        self.stations.append(station)

    def add_track(self, track):
        self.tracks.append(track)

    def add_train(self, train):
        self.trains.append(train)

    def add_route(self, route_id, stations_list):
        self.routes[route_id] = stations_list

    def get_station_by_id(self, station_id):
        for station in self.stations:
            if station.station_id == station_id:
                return station
        return None

    def get_track_by_id(self, track_id):
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def get_train_by_no(self, train_no):
        for train in self.trains:
            if train.train_no == train_no:
                return train
        return None
