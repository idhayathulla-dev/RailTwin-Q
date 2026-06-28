import json
import os
from models.station import Station
from models.track import Track
from models.train import Train
from models.railway_network import RailwayNetwork

class DataLoader:
    @staticmethod
    def load_network(data_dir="data") -> RailwayNetwork:
        network = RailwayNetwork()

        # Load stations
        stations_path = os.path.join(data_dir, "stations.json")
        if os.path.exists(stations_path):
            with open(stations_path, "r") as f:
                data = json.load(f)
                for item in data:
                    station = Station(
                        station_id=item["station_id"],
                        name=item["name"],
                        latitude=item["latitude"],
                        longitude=item["longitude"],
                        platforms=item["platforms"]
                    )
                    network.add_station(station)

        # Load tracks
        tracks_path = os.path.join(data_dir, "tracks.json")
        if os.path.exists(tracks_path):
            with open(tracks_path, "r") as f:
                data = json.load(f)
                for item in data:
                    track = Track(
                        track_id=item["id"],
                        source_station_id=item["source_station_id"],
                        destination_station_id=item["destination_station_id"],
                        distance=item["distance"],
                        max_speed=item["max_speed"],
                        capacity=item.get("capacity", 5)
                    )
                    network.add_track(track)

        # Load routes
        routes_path = os.path.join(data_dir, "routes.json")
        if os.path.exists(routes_path):
            with open(routes_path, "r") as f:
                data = json.load(f)
                for item in data:
                    network.add_route(
                        route_id=item["route_id"],
                        stations_list=item["stations"]
                    )

        # Load trains
        trains_path = os.path.join(data_dir, "trains.json")
        if os.path.exists(trains_path):
            with open(trains_path, "r") as f:
                data = json.load(f)
                for item in data:
                    train_type = item.get("train_type")
                    if not train_type:
                        name_lower = item["name"].lower()
                        if "superfast" in name_lower or "mail" in name_lower or "shatabdi" in name_lower or "rajdhani" in name_lower:
                            train_type = "SUPERFAST"
                        elif "express" in name_lower:
                            train_type = "EXPRESS"
                        elif "goods" in name_lower or "freight" in name_lower or "coal" in name_lower:
                            train_type = "FREIGHT"
                        else:
                            train_type = "PASSENGER"

                    max_speed = item.get("max_speed")
                    if not max_speed:
                        if train_type == "SUPERFAST":
                            max_speed = 130
                        elif train_type == "EXPRESS":
                            max_speed = 110
                        elif train_type == "FREIGHT":
                            max_speed = 75
                        else:
                            max_speed = 90

                    train = Train(
                        train_no=item["train_no"],
                        name=item["name"],
                        route_id=item["route_id"],
                        current_station_id=item["current_station_id"],
                        speed=item["speed"],
                        delay=item["delay"],
                        train_type=train_type,
                        max_speed=max_speed
                    )
                    network.add_train(train)

        return network
