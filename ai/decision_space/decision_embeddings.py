import json
import os

class DecisionEmbeddingGenerator:
    @staticmethod
    def generate_embeddings(actions: list, cost_vectors: list, passenger_impacts: dict, robustness_data: dict, data_dir="datasets") -> dict:
        """
        Converts every candidate action into a numerical optimization vector:
        [delay_benefit, energy_cost, risk_cost, passenger_benefit, schedule_stability, confidence_score, robustness_score]
        """
        embeddings = {}

        for idx, act in enumerate(actions):
            action_id = idx + 1
            act_type = act.get("action")

            # Extract metrics
            cost_vector = next((cv["cost_vector"] for cv in cost_vectors if cv["action_id"] == action_id), {})
            p_impact = passenger_impacts.get(str(action_id), {})
            r_data = robustness_data.get(str(action_id), {})

            # Map to 7-dimensional numerical vector
            vec = [
                abs(cost_vector.get("delay", 0.0)),  # delay benefit (reduction)
                cost_vector.get("energy", 0.0),      # energy cost
                abs(cost_vector.get("risk", 0.0)),    # risk mitigation
                p_impact.get("passengers_saved", 0) / 1000.0, # passenger benefit
                cost_vector.get("schedule_stability", 0.0),  # schedule stability impact
                act.get("confidence", 0.9),          # confidence
                r_data.get("robustness_score", 100) / 100.0 # robustness score
            ]

            embeddings[str(action_id)] = {
                "action_id": action_id,
                "action": act_type,
                "embedding_vector": [round(float(v), 3) for v in vec]
            }

        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "decision_vectors.json"), "w", encoding="utf-8") as f:
            json.dump(embeddings, f, indent=4)

        return embeddings
