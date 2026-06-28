import networkx as nx

class FeatureEngineering:
    @staticmethod
    def compute_graph_metrics(graph: nx.Graph) -> dict:
        """
        Computes NetworkX graph topology centralities for all stations in the network.
        Returns a dictionary mapping station_id to its centrality values.
        """
        metrics = {}
        
        # Centralities
        try:
            betweenness = nx.betweenness_centrality(graph, weight="weight")
        except Exception:
            betweenness = {node: 0.0 for node in graph.nodes}
            
        try:
            closeness = nx.closeness_centrality(graph, distance="weight")
        except Exception:
            closeness = {node: 0.0 for node in graph.nodes}

        # Calculate degree and local connectivity for each node
        for node in graph.nodes:
            deg = graph.degree(node)
            
            # Station connectivity: we define it as average neighbor degree
            neighbors = list(graph.neighbors(node))
            if neighbors:
                avg_neighbor_deg = sum(graph.degree(n) for n in neighbors) / len(neighbors)
            else:
                avg_neighbor_deg = 0.0
                
            metrics[node] = {
                "node_degree": deg,
                "betweenness_centrality": round(betweenness.get(node, 0.0), 4),
                "closeness_centrality": round(closeness.get(node, 0.0), 4),
                "station_connectivity": round(avg_neighbor_deg, 4)
            }
            
        return metrics

    @staticmethod
    def get_time_of_day(time_str: str) -> str:
        """
        Classifies time string ("HH:MM") into MORNING, AFTERNOON, EVENING, or NIGHT.
        """
        try:
            hrs, _ = map(int, time_str.split(":"))
            if 6 <= hrs < 12:
                return "MORNING"
            elif 12 <= hrs < 17:
                return "AFTERNOON"
            elif 17 <= hrs < 21:
                return "EVENING"
            else:
                return "NIGHT"
        except Exception:
            return "MORNING"

    @staticmethod
    def calculate_station_congestion(platforms_occupied: int, trains_waiting: int, platforms_total: int) -> float:
        """
        Computes station congestion score between 0.0 and 100.0.
        """
        if platforms_total <= 0:
            return 0.0
        # Congestion score can exceed 100 if waiting trains queue up outside the platforms
        score = ((platforms_occupied + trains_waiting) / platforms_total) * 100.0
        return round(score, 2)

    @staticmethod
    def calculate_network_metrics(network_state: dict) -> dict:
        """
        Computes aggregated network performance scores (utilizations, congestion)
        from a single tick's state dictionary.
        """
        # Platform utilization
        total_platforms = sum(s["platforms_total"] for s in network_state["stations"])
        used_platforms = sum(s["platforms_occupied"] for s in network_state["stations"])
        platform_util = (used_platforms / total_platforms * 100.0) if total_platforms > 0 else 0.0

        # Track utilization
        total_capacity = sum(t["track_capacity"] for t in network_state["tracks"])
        used_capacity = sum(t["trains_on_track"] for t in network_state["tracks"])
        track_util = (used_capacity / total_capacity * 100.0) if total_capacity > 0 else 0.0

        # Network congestion score: combination of platform and track utilizations
        congestion_score = (platform_util * 0.4) + (track_util * 0.6)
        
        return {
            "platform_utilization_percent": round(platform_util, 2),
            "track_utilization_percent": round(track_util, 2),
            "network_congestion_score": round(congestion_score, 2)
        }
