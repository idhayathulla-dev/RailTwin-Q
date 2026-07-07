import json
import os

class CounterfactualEngine:
    @staticmethod
    def generate_counterfactuals(actions: list, baseline_recovery_time: int, data_dir="datasets") -> dict:
        """
        Generates what-if operational comparisons for all candidate actions.
        """
        scenarios = {
            "Scenario A (No Action)": {
                "recovery_time_mins": baseline_recovery_time,
                "delay_reduction_minutes": 0.0,
                "congestion_reduction_percent": 0.0,
                "affected_trains_reduced": 0,
                "affected_stations_reduced": 0,
                "csi_improvement": 0.0
            }
        }

        # Platform Swap scenario
        swap_saving = sum([a["expected_delay_reduction"] for a in actions if a["action"] == "PLATFORM_SWAP"])
        scenarios["Scenario B (Platform Swap)"] = {
            "recovery_time_mins": max(10, baseline_recovery_time - int(swap_saving * 0.8)),
            "delay_reduction_minutes": round(swap_saving, 1),
            "congestion_reduction_percent": round(swap_saving * 2.0, 1),
            "affected_trains_reduced": 1 if swap_saving > 0 else 0,
            "affected_stations_reduced": 1 if swap_saving > 0 else 0,
            "csi_improvement": round(swap_saving * 1.5, 1)
        }

        # Speed Adjustment scenario
        speed_saving = sum([a["expected_delay_reduction"] for a in actions if a["action"] == "SPEED_ADJUST"])
        scenarios["Scenario C (Speed Adjustment)"] = {
            "recovery_time_mins": max(10, baseline_recovery_time - int(speed_saving * 0.8)),
            "delay_reduction_minutes": round(speed_saving, 1),
            "congestion_reduction_percent": round(speed_saving * 1.2, 1),
            "affected_trains_reduced": 1 if speed_saving > 0 else 0,
            "affected_stations_reduced": 0,
            "csi_improvement": round(speed_saving * 1.0, 1)
        }

        # Reroute scenario
        reroute_saving = sum([a["expected_delay_reduction"] for a in actions if a["action"] == "REROUTE"])
        scenarios["Scenario D (Reroute)"] = {
            "recovery_time_mins": max(10, baseline_recovery_time - int(reroute_saving * 0.9)),
            "delay_reduction_minutes": round(reroute_saving, 1),
            "congestion_reduction_percent": round(reroute_saving * 2.5, 1),
            "affected_trains_reduced": 2 if reroute_saving > 0 else 0,
            "affected_stations_reduced": 1 if reroute_saving > 0 else 0,
            "csi_improvement": round(reroute_saving * 2.0, 1)
        }

        # Hold Train scenario
        hold_saving = sum([a["expected_delay_reduction"] for a in actions if a["action"] == "HOLD"])
        scenarios["Scenario E (Hold Train)"] = {
            "recovery_time_mins": max(10, baseline_recovery_time - int(hold_saving * 0.5)),
            "delay_reduction_minutes": round(hold_saving, 1),
            "congestion_reduction_percent": round(hold_saving * 0.8, 1),
            "affected_trains_reduced": 1 if hold_saving > 0 else 0,
            "affected_stations_reduced": 0,
            "csi_improvement": round(hold_saving * 0.5, 1)
        }

        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "counterfactual_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=4)

        return scenarios
