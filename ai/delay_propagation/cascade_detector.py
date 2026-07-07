import networkx as nx
import numpy as np

class CascadeDetector:
    @staticmethod
    def detect_cascading_disruptions(G: nx.DiGraph, horizons=[15, 30, 60]) -> dict:
        """
        Simulates propagation of delays along operational dependency edges up to 3 hops.
        Tracks depth, propagation probabilities, and decays confidence levels.
        """
        cascade_state = {}

        # Locate seed delay nodes (trains with current delay > 5 minutes)
        seed_nodes = []
        for node, attrs in G.nodes(data=True):
            if attrs.get("type") == "train" and attrs.get("current_delay", 0.0) > 5.0:
                # Seed confidence defaults to delay predictor confidence (e.g. 96%)
                seed_conf = attrs.get("confidence", 96.0)
                seed_nodes.append((node, attrs.get("current_delay"), 1.0, seed_conf))

        for horizon in horizons:
            affected_entities = {}
            max_depth = 0
            total_delay_increase = 0.0
            
            # Map of node -> (accumulated_delay, prob_affected, hop_count, confidence)
            visited = {}
            for seed, d, p, conf in seed_nodes:
                visited[seed] = (d, p, 0, conf)
            
            # Simple queue for propagation simulation
            queue = [seed for seed, _, _, _ in seed_nodes]
            
            while queue:
                curr = queue.pop(0)
                curr_delay, curr_prob, curr_hops, curr_conf = visited[curr]
                
                # Capped at 3 hops (multi-hop constraints)
                if curr_hops >= 3:
                    continue
                    
                # Check adjacent neighbors
                for neighbor in G.neighbors(curr):
                    edge = G[curr][neighbor]
                    
                    p_trans = edge.get("propagation_probability", 0.5)
                    d_trans = edge.get("expected_delay_transfer", 5.0)
                    t_trans = edge.get("transfer_time", 10)
                    
                    # Horizon constraints: if transfer time exceeds horizon, cascade hasn't reached neighbor
                    if curr_hops * 10 + t_trans > horizon:
                        continue
                        
                    new_prob = curr_prob * p_trans
                    # Skip if probability is too low to represent a realistic threat
                    if new_prob < 0.10:
                        continue
                        
                    # Calculate parent uncertainty
                    parent_uncertainty = 1.0 - (curr_conf / 100.0)
                    # Propagate confidence downstream
                    new_conf = round(float(curr_conf * p_trans * (1.0 - parent_uncertainty * 0.5)), 1)
                    new_conf = max(10.0, min(98.0, new_conf))
                    
                    new_delay = curr_delay * 0.7 + d_trans * new_prob
                    new_hops = curr_hops + 1
                    
                    if neighbor not in visited or visited[neighbor][0] < new_delay:
                        visited[neighbor] = (new_delay, new_prob, new_hops, new_conf)
                        if neighbor not in queue:
                            queue.append(neighbor)
                            
                        # Record attributes
                        n_attrs = G.nodes[neighbor]
                        n_type = n_attrs.get("type")
                        
                        # Calculate confidence intervals
                        half_width = round(1.96 * (1.0 - new_conf / 100.0) * 15.0, 1)
                        conf_interval = [max(0.0, round(new_delay - half_width, 1)), round(new_delay + half_width, 1)]
                        
                        entity_key = neighbor
                        affected_entities[entity_key] = {
                            "type": n_type,
                            "id": neighbor.split("_")[-1],
                            "name": n_attrs.get("name", neighbor),
                            "predicted_value": round(new_delay, 1),
                            "propagation_depth": new_hops,
                            "confidence_score": new_conf,
                            "confidence_interval": conf_interval,
                            "uncertainty_score": round(1.0 - new_conf / 100.0, 3)
                        }
                        
                        max_depth = max(max_depth, new_hops)
                        total_delay_increase += d_trans

            # Group affected entities for output compatibility
            aff_trains = [int(val["id"]) for k, val in affected_entities.items() if val["type"] == "train"]
            aff_stations = [int(val["id"]) for k, val in affected_entities.items() if val["type"] == "station"]
            aff_tracks = [val["id"] for k, val in affected_entities.items() if val["type"] == "track"]

            cascade_state[horizon] = {
                "affected_trains": aff_trains,
                "affected_stations": aff_stations,
                "affected_tracks": aff_tracks,
                "max_depth": max_depth,
                "average_delay_increase": round(total_delay_increase / max(1, len(visited) - len(seed_nodes)), 1),
                "detailed_entities": affected_entities
            }

        return cascade_state
