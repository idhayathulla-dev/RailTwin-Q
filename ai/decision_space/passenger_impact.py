import json
import os

class PassengerImpactEngine:
    @staticmethod
    def calculate_impacts(actions: list, data_dir="datasets") -> dict:
        """
        Estimates passenger-level impacts based on train classifications.
        """
        impacts = {}

        for idx, act in enumerate(actions):
            action_id = idx + 1
            act_type = act.get("action")
            train_name = act.get("train_name", "")
            
            # Determine passenger volume based on train categories
            if "Mail" in train_name or "Express" in train_name or "Kerala" in train_name:
                base_passengers = 1000  # Express category
            elif "Sapthagiri" in train_name:
                base_passengers = 700   # Passenger category
            else:
                base_passengers = 400   # MEMU / local commuter category

            saving = act.get("expected_delay_reduction", 5.0)

            # Metrics
            if act_type == "PLATFORM_SWAP":
                delayed = int(base_passengers * 0.1)
                saved = int(base_passengers * 0.9)
                missed_avoided = 5
                crowd_reduction = 25.0
                platform_reduction = 35.0
                wait_reduction = int(saving * 0.8)
            elif act_type == "REROUTE":
                delayed = int(base_passengers * 0.05)
                saved = int(base_passengers * 0.95)
                missed_avoided = 12
                crowd_reduction = 45.0
                platform_reduction = 15.0
                wait_reduction = int(saving * 0.95)
            elif act_type == "HOLD":
                delayed = int(base_passengers * 0.6)  # High penalty for held train
                saved = int(base_passengers * 0.4)
                missed_avoided = -3
                crowd_reduction = -10.0
                platform_reduction = -15.0
                wait_reduction = int(saving * 0.3)
            elif act_type == "SPEED_ADJUST":
                delayed = 0
                saved = base_passengers
                missed_avoided = 2
                crowd_reduction = 5.0
                platform_reduction = 5.0
                wait_reduction = int(saving)
            else:
                delayed = 0
                saved = 0
                missed_avoided = 0
                crowd_reduction = 0.0
                platform_reduction = 0.0
                wait_reduction = 0

            impacts[str(action_id)] = {
                "action_id": action_id,
                "action": act_type,
                "passengers_delayed": delayed,
                "passengers_saved": saved,
                "missed_connections_avoided": missed_avoided,
                "station_crowd_reduction_percent": crowd_reduction,
                "platform_crowd_reduction_percent": platform_reduction,
                "passenger_waiting_time_reduction_mins": wait_reduction
            }

        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "passenger_impact.json"), "w", encoding="utf-8") as f:
            json.dump(impacts, f, indent=4)

        return impacts
