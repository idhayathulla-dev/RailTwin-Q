import json
import os

class OperationalCostEngine:
    @staticmethod
    def generate_costs(actions: list, data_dir="datasets") -> list:
        """
        Calculates a complete operational cost vector for each candidate action.
        All values are normalized between 0.0 and 1.0.
        """
        cost_vectors = []

        for idx, act in enumerate(actions):
            action_id = idx + 1
            act_type = act.get("action")

            # Cost Vector Templates (Normalized 0 to 1)
            cost = {
                "action_id": action_id,
                "action": act_type,
                "cost_vector": {
                    "delay_saved": 0.0,
                    "passenger_delay": 0.0,
                    "energy_consumption": 0.0,
                    "fuel_cost": 0.0,
                    "crew_impact": 0.0,
                    "track_utilization": 0.0,
                    "platform_usage": 0.0,
                    "maintenance_impact": 0.0,
                    "schedule_stability": 0.0,
                    "operational_complexity": 0.0,
                    "safety_risk": 0.0
                }
            }

            vec = cost["cost_vector"]

            if act_type == "PLATFORM_SWAP":
                vec["delay_saved"] = -0.4  # Negative indicates improvement
                vec["passenger_delay"] = 0.1
                vec["energy_consumption"] = 0.05
                vec["fuel_cost"] = 0.05
                vec["crew_impact"] = 0.3
                vec["track_utilization"] = 0.1
                vec["platform_usage"] = 0.6  # High platform slot occupation
                vec["maintenance_impact"] = 0.0
                vec["schedule_stability"] = 0.15
                vec["operational_complexity"] = 0.4
                vec["safety_risk"] = 0.1
            elif act_type == "REROUTE":
                vec["delay_saved"] = -0.7
                vec["passenger_delay"] = 0.3
                vec["energy_consumption"] = 0.7  # High energy index due to bypass detour
                vec["fuel_cost"] = 0.7
                vec["crew_impact"] = 0.5
                vec["track_utilization"] = 0.5
                vec["platform_usage"] = 0.2
                vec["maintenance_impact"] = 0.2
                vec["schedule_stability"] = 0.8  # Major schedule disruption
                vec["operational_complexity"] = 0.7
                vec["safety_risk"] = 0.2
            elif act_type == "HOLD":
                vec["delay_saved"] = -0.3
                vec["passenger_delay"] = 0.6  # Passenger delay increases
                vec["energy_consumption"] = 0.0  # Saves fuel by standing still
                vec["fuel_cost"] = 0.0
                vec["crew_impact"] = 0.4
                vec["track_utilization"] = 0.2
                vec["platform_usage"] = 0.5
                vec["maintenance_impact"] = 0.0
                vec["schedule_stability"] = 0.5
                vec["operational_complexity"] = 0.3
                vec["safety_risk"] = 0.05
            elif act_type == "SPEED_ADJUST":
                vec["delay_saved"] = -0.2
                vec["passenger_delay"] = 0.1
                vec["energy_consumption"] = 0.4
                vec["fuel_cost"] = 0.4
                vec["crew_impact"] = 0.1
                vec["track_utilization"] = 0.3
                vec["platform_usage"] = 0.1
                vec["maintenance_impact"] = 0.1
                vec["schedule_stability"] = 0.2
                vec["operational_complexity"] = 0.2
                vec["safety_risk"] = 0.15

            # Compatibility mapping
            vec["delay"] = vec["delay_saved"] * 20.0
            vec["risk"] = vec["safety_risk"] * 50.0
            vec["energy"] = vec["energy_consumption"] * 10.0
            vec["operational_cost"] = vec["operational_complexity"] * 5.0

            cost_vectors.append(cost)

        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "cost_vector.json"), "w", encoding="utf-8") as f:
            json.dump(cost_vectors, f, indent=4)

        return cost_vectors
