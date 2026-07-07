import json
import os

class RobustnessEngine:
    @staticmethod
    def evaluate_robustness(actions: list, data_dir="datasets") -> dict:
        """
        Evaluates each candidate action across multiple failure profiles.
        """
        profiles = ["Heavy Rain", "Signal Failure", "Track Blockage", "Maintenance", "Festival Rush", "Power Failure"]
        robustness_data = {}

        for idx, act in enumerate(actions):
            action_id = idx + 1
            act_type = act.get("action")

            # Determine compatibility indicators
            compatibility = {p: True for p in profiles}

            if act_type == "PLATFORM_SWAP":
                compatibility["Signal Failure"] = False  # Swap depends on signal routing alignment
                compatibility["Power Failure"] = False
                score = 66
            elif act_type == "REROUTE":
                compatibility["Track Blockage"] = False  # Bypassing may be blocked
                compatibility["Signal Failure"] = False
                score = 66
            elif act_type == "HOLD":
                compatibility["Festival Rush"] = False  # Holding causes massive platform crowding
                score = 83
            elif act_type == "SPEED_ADJUST":
                compatibility["Power Failure"] = False
                score = 83
            else:
                score = 100

            robustness_data[str(action_id)] = {
                "action_id": action_id,
                "action": act_type,
                "compatibility": compatibility,
                "robustness_score": score
            }

        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "robustness_report.json"), "w", encoding="utf-8") as f:
            json.dump(robustness_data, f, indent=4)

        return robustness_data
