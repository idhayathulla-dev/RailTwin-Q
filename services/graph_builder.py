import networkx as nx
from models.railway_network import RailwayNetwork

class GraphBuilder:
    @staticmethod
    def build_graph(network: RailwayNetwork) -> nx.Graph:
        """
        Builds a NetworkX undirected graph representing the railway network.
        Stations are nodes, and Tracks are edges.
        """
        G = nx.Graph()

        # Add stations as nodes
        for station in network.stations:
            G.add_node(
                station.station_id,
                name=station.name,
                latitude=station.latitude,
                longitude=station.longitude,
                platforms=station.platforms,
                station_obj=station
            )

        # Add tracks as edges
        for track in network.tracks:
            # We calculate weight as a function of travel time: distance / max_speed
            # (distance in km, speed in km/h)
            travel_time_hours = track.distance / track.max_speed if track.max_speed > 0 else float('inf')
            G.add_edge(
                track.source_station_id,
                track.destination_station_id,
                track_id=track.track_id,
                distance=track.distance,
                max_speed=track.max_speed,
                capacity=track.capacity,
                weight=travel_time_hours,
                track_obj=track
            )

        return G
