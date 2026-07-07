import networkx as nx

class CandidateActionGenerator:
    @staticmethod
    def generate_candidate_actions(G: nx.DiGraph, network) -> list:
        """
        Generates candidate operational actions based on active disruptions and bottleneck nodes.
        These actions form the search space for Layer 5 (Quantum Optimization).
        """
        actions = []

        # 1. Platform Swaps (triggered by station platform saturation)
        for s in network.stations:
            if s.station_congestion_score >= 40.0:
                actions.append({
                    "action": "PLATFORM_SWAP",
                    "station_id": s.station_id,
                    "station_name": s.name,
                    "expected_congestion_reduction": round(float(s.station_congestion_score * 0.3), 1),
                    "expected_delay_reduction": 8.5,
                    "confidence": 0.88,
                    "passenger_impact": "Low"
                })

        # 2. Rerouting (triggered by track saturation/blockages)
        for tr in network.tracks:
            if tr.occupancy_percent >= 50.0 or tr.blocked:
                # Find trains heading to or moving on this track
                moving_trains = [t for t in network.trains if t.current_track_id == tr.track_id and t.status == "MOVING"]
                for mt in moving_trains:
                    # Alternate routes are available if we bypass Katpadi or use alternate junctions (e.g. Track 2 bypass)
                    actions.append({
                        "action": "REROUTE",
                        "train_id": mt.train_no,
                        "train_name": mt.name,
                        "track_id": tr.track_id,
                        "expected_delay_reduction": 14.2,
                        "confidence": 0.91,
                        "passenger_impact": "Medium"
                    })

        # 3. Hold Upstream (triggered by leading delays)
        delayed_trains = [t for t in network.trains if t.delay > 10.0]
        for dt in delayed_trains:
            # Propose holding trailing trains to avoid safety block accumulation
            actions.append({
                "action": "HOLD",
                "train_id": dt.train_no,
                "train_name": dt.name,
                "duration_mins": 5,
                "expected_delay_reduction": 6.0,
                "confidence": 0.85,
                "passenger_impact": "Low"
            })

        # 4. Speed Adjustment
        for t in network.trains:
            if t.status == "MOVING" and t.delay > 5.0:
                actions.append({
                    "action": "SPEED_ADJUST",
                    "train_id": t.train_no,
                    "train_name": t.name,
                    "new_speed_kmp": 75 if t.speed > 80 else 90,
                    "expected_delay_reduction": 4.5,
                    "confidence": 0.93,
                    "passenger_impact": "None"
                })

        # Fallback default actions if network is completely stable
        if not actions:
            actions.append({
                "action": "SCHEDULE_MAINTENANCE",
                "expected_delay_reduction": 0.0,
                "confidence": 0.99,
                "passenger_impact": "None"
            })

        return actions

    @staticmethod
    def compare_scenarios(actions: list, baseline_recovery_time: int) -> list:
        """
        Simulates multiple intervention scenarios and estimates the expected recovery time for each.
        """
        scenarios = [
            {
                "scenario_id": "Scenario A",
                "name": "No Action (Baseline)",
                "expected_recovery_time_mins": baseline_recovery_time,
                "remaining_disruption_mins": baseline_recovery_time,
                "expected_global_delay_reduction": 0.0,
                "confidence": 0.95
            }
        ]

        # Scenario B (Platform Swap actions applied)
        swap_reduction = sum([a["expected_delay_reduction"] for a in actions if a["action"] == "PLATFORM_SWAP"])
        rec_time_b = max(10, baseline_recovery_time - int(swap_reduction))
        scenarios.append({
            "scenario_id": "Scenario B",
            "name": "Junction Platform Swaps",
            "expected_recovery_time_mins": rec_time_b,
            "remaining_disruption_mins": rec_time_b,
            "expected_global_delay_reduction": round(swap_reduction, 1),
            "confidence": 0.88
        })

        # Scenario C (Reroute / Speed limit actions applied)
        reroute_reduction = sum([a["expected_delay_reduction"] for a in actions if a["action"] in ["REROUTE", "SPEED_ADJUST"]])
        rec_time_c = max(10, baseline_recovery_time - int(reroute_reduction))
        scenarios.append({
            "scenario_id": "Scenario C",
            "name": "Dynamic Reroutes & Speed Limits",
            "expected_recovery_time_mins": rec_time_c,
            "remaining_disruption_mins": rec_time_c,
            "expected_global_delay_reduction": round(reroute_reduction, 1),
            "confidence": 0.91
        })

        return scenarios
