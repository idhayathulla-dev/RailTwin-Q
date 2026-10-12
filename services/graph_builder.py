"""Graph builder for the RailTwin-Q railway network.

This module exposes :class:`GraphBuilder`, which converts a
:class:`models.railway_network.RailwayNetwork` into a NetworkX undirected
graph.  Each station becomes a node and each track becomes an edge, with
the domain objects attached as node/edge attributes so downstream
analysis can walk back to the originals.
"""

from __future__ import annotations

import math

import networkx as nx

from models.railway_network import RailwayNetwork


class GraphBuilder:
    """Builds NetworkX graphs from a :class:`RailwayNetwork` instance."""

    @staticmethod
    def build_graph(network: RailwayNetwork) -> nx.Graph:
        """Build an undirected NetworkX graph from *network*.

        Stations are added as nodes and tracks as edges.  In addition to
        the scalar geometry/speed fields, the raw domain objects are
        stashed on each node/edge as ``station_obj`` / ``track_obj`` so
        that consumers can recover full state without a second lookup.

        Args:
            network: The railway network whose stations and tracks should
                be represented.

        Returns:
            A :class:`networkx.Graph` with node and edge attributes
            derived from the given *network*.
        """
        graph: nx.Graph = nx.Graph()

        # Add stations as nodes.
        for station in network.stations:
            graph.add_node(
                station.station_id,
                name=station.name,
                code=getattr(station, "code", f"ST{station.station_id}"),
                x=getattr(station, "x", 100),
                y=getattr(station, "y", 200),
                latitude=station.latitude,
                longitude=station.longitude,
                platforms=station.platforms,
                station_type=getattr(station, "station_type", "REGULAR"),
                is_junction=getattr(station, "is_junction", False),
                station_obj=station,
            )

        # Add tracks as edges.
        for track in network.tracks:
            weight = track.distance / track.max_speed if track.max_speed > 0 else math.inf
            graph.add_edge(
                track.source_station_id,
                track.destination_station_id,
                track_id=track.track_id,
                name=getattr(track, "name", f"Track {track.track_id}"),
                distance=track.distance,
                max_speed=track.max_speed,
                capacity=track.capacity,
                track_type=getattr(track, "track_type", "DOUBLE_TRACK"),
                direction=getattr(track, "direction", "BOTH"),
                weight=weight,
                track_obj=track,
            )

        return graph
