import networkx as nx

class CriticalityEngine:
    @staticmethod
    def calculate_criticality_scores(G: nx.DiGraph, network) -> tuple:
        """
        Calculates dynamic criticality scores (0-100) for every node in the propagation DAG.
        Combines degree centrality, delay levels, platform queues, and alternative track availability.
        """
        deg_centrality = nx.degree_centrality(G)

        train_ranks = []
        station_ranks = []
        track_ranks = []

        for node, attrs in G.nodes(data=True):
            n_type = attrs.get("type")
            node_deg = deg_centrality.get(node, 0.0)
            influence_score = node_deg * 40.0

            if n_type == "train":
                train_no = attrs.get("train_no")
                delay = attrs.get("current_delay", 0.0)
                priority = attrs.get("priority", 0)
                
                crit = min(100.0, influence_score + delay * 1.2 + priority * 10.0 + attrs.get("remaining_stations", 0) * 3.0)
                train_ranks.append({
                    "id": train_no,
                    "name": attrs.get("name"),
                    "criticality_score": round(crit, 1),
                    "delay": delay
                })

            elif n_type == "station":
                s_id = attrs.get("station_id")
                congestion = attrs.get("current_congestion", 0.0)
                waiting = attrs.get("trains_waiting", 0)
                
                crit = min(100.0, influence_score + congestion * 0.5 + waiting * 10.0)
                station_ranks.append({
                    "id": s_id,
                    "name": attrs.get("name"),
                    "criticality_score": round(crit, 1),
                    "congestion": congestion
                })

            elif n_type == "track":
                tr_id = attrs.get("track_id")
                occupancy = attrs.get("current_occupancy", 0.0)
                blocked = attrs.get("blocked", False)
                
                # Check alternative routing availability (track 2 has alternate bypass, tracks 1 & 3 do not)
                alt_penalty = 0.0 if tr_id == "2" else 20.0
                blockage_penalty = 40.0 if blocked else 0.0
                
                crit = min(100.0, influence_score + occupancy * 0.4 + alt_penalty + blockage_penalty)
                track_ranks.append({
                    "id": tr_id,
                    "criticality_score": round(crit, 1),
                    "occupancy": occupancy,
                    "status": "BLOCKED" if blocked else "NORMAL"
                })

        # Sort ranks by criticality descending
        train_ranks = sorted(train_ranks, key=lambda x: x["criticality_score"], reverse=True)
        station_ranks = sorted(station_ranks, key=lambda x: x["criticality_score"], reverse=True)
        track_ranks = sorted(track_ranks, key=lambda x: x["criticality_score"], reverse=True)

        return train_ranks, station_ranks, track_ranks

    @staticmethod
    def calculate_risk_scores(G: nx.DiGraph, network) -> dict:
        """
        Computes dynamic operational Risk Scores (0-100) for every node.
        Risk is a function of: Probability of Disruption, Propagation Potential, and Recovery Difficulty.
        """
        deg_centrality = nx.degree_centrality(G)
        
        train_risks = []
        station_risks = []
        track_risks = []

        for node, attrs in G.nodes(data=True):
            n_type = attrs.get("type")
            deg = deg_centrality.get(node, 0.0)
            
            # Risk components
            prob_disruption = 0.0
            propagation_potential = deg * 50.0
            recovery_difficulty = 0.0
            
            if n_type == "train":
                t_no = attrs.get("train_no")
                delay = attrs.get("current_delay", 0.0)
                prob_disruption = min(100.0, delay * 2.0)
                recovery_difficulty = 20.0 if attrs.get("priority", 0) > 0 else 10.0
                
                risk = 0.3 * prob_disruption + 0.3 * propagation_potential + 0.4 * recovery_difficulty
                train_risks.append({
                    "id": t_no,
                    "name": attrs.get("name"),
                    "risk_score": round(min(100.0, risk), 1)
                })
                
            elif n_type == "station":
                s_id = attrs.get("station_id")
                congestion = attrs.get("current_congestion", 0.0)
                waiting = attrs.get("trains_waiting", 0)
                prob_disruption = congestion
                recovery_difficulty = waiting * 15.0
                
                risk = 0.3 * prob_disruption + 0.3 * propagation_potential + 0.4 * recovery_difficulty
                station_risks.append({
                    "id": s_id,
                    "name": attrs.get("name"),
                    "risk_score": round(min(100.0, risk), 1)
                })
                
            elif n_type == "track":
                tr_id = attrs.get("track_id")
                occupancy = attrs.get("current_occupancy", 0.0)
                blocked = attrs.get("blocked", False)
                prob_disruption = occupancy
                recovery_difficulty = 40.0 if blocked else 10.0
                
                risk = 0.3 * prob_disruption + 0.3 * propagation_potential + 0.4 * recovery_difficulty
                track_risks.append({
                    "id": tr_id,
                    "risk_score": round(min(100.0, risk), 1)
                })

        train_risks = sorted(train_risks, key=lambda x: x["risk_score"], reverse=True)
        station_risks = sorted(station_risks, key=lambda x: x["risk_score"], reverse=True)
        track_risks = sorted(track_risks, key=lambda x: x["risk_score"], reverse=True)

        return {
            "train_risks": train_risks,
            "station_risks": station_risks,
            "track_risks": track_risks
        }
