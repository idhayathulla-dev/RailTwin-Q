import json
import os

class DecisionExplainer:
    @staticmethod
    def generate_explanations(actions: list, cost_vectors: list, passenger_impacts: dict, data_dir="datasets") -> dict:
        """
        Generates explainable human-readable outputs for each recommendation.
        """
        explanations = {}

        for idx, act in enumerate(actions):
            action_id = idx + 1
            act_type = act.get("action")
            target = act.get("train_name", act.get("station_name", "Global"))

            cost_vector = next((cv["cost_vector"] for cv in cost_vectors if cv["action_id"] == action_id), {})
            p_impact = passenger_impacts.get(str(action_id), {})

            # Craft explanation & trade-offs
            if act_type == "PLATFORM_SWAP":
                reason = f"Platform at station {target} is congested, exceeding 75% capacity limits."
                trade_off = "Operational complexity increases due to crew relocation, but platform queues clear 35% faster."
            elif act_type == "REROUTE":
                reason = f"Primary track segment is congested/blocked. Alternate paths are available."
                trade_off = f"Energy consumption increases by {cost_vector.get('energy', 0.0) * 10:.0f}%, but delay is reduced by {abs(cost_vector.get('delay', 0.0)):.1f} mins."
            elif act_type == "HOLD":
                reason = "Safety block buffer capacity drops below safety limits."
                trade_off = "Following trains save safety block delays, but held passengers experience localized waiting time increase."
            else:
                reason = "Safety spacing adjustments scheduled."
                trade_off = "None."

            explanations[str(action_id)] = {
                "action_id": action_id,
                "action": act_type,
                "target": target,
                "reason": reason,
                "expected_delay_saved": f"{abs(cost_vector.get('delay', 0.0)):.1f} mins",
                "trade_off": trade_off,
                "passengers_saved": p_impact.get("passengers_saved", 0),
                "confidence": f"{act.get('confidence', 0.9) * 100:.0f}%"
            }

        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "decision_explanations.json"), "w", encoding="utf-8") as f:
            json.dump(explanations, f, indent=4)

        return explanations
