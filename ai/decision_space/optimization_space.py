import json
import os

class OptimizationSearchSpaceGenerator:
    @staticmethod
    def compile_search_space(
        actions: list,
        reasoning: dict,
        counterfactuals: dict,
        cost_vectors: list,
        passenger_impacts: dict,
        robustness: dict,
        pareto_front: dict,
        embeddings: dict,
        explanations: dict,
        dependency_graph: dict,
        constraints: dict,
        data_dir="datasets"
    ) -> dict:
        """
        Compiles the complete mathematical optimization search space for Layer 5 (Quantum Optimization).
        """
        search_space = {
            "decision_variables": {
                str(idx + 1): {
                    "action_id": idx + 1,
                    "action": act.get("action"),
                    "target": act.get("train_name", act.get("station_name", "Global")),
                    "feasible": True,
                    "variable_symbol": f"x_{idx + 1}"
                } for idx, act in enumerate(actions)
            },
            "constraints": constraints,
            "cost_vectors": cost_vectors,
            "expected_rewards": {
                str(cv["action_id"]): {
                    "delay_reduction_mins": abs(cv["cost_vector"]["delay"]),
                    "passenger_saved_count": passenger_impacts.get(str(cv["action_id"]), {}).get("passengers_saved", 0),
                    "robustness_rating": robustness.get(str(cv["action_id"]), {}).get("robustness_score", 100)
                } for cv in cost_vectors
            },
            "dependencies": dependency_graph,
            "pareto_optimal_front": pareto_front,
            "decision_embeddings": embeddings,
            "confidence_intervals": {
                str(idx + 1): {
                    "confidence_score": act.get("confidence", 0.9),
                    "best_case_delay_saved": round(act.get("expected_delay_reduction", 5.0) * 1.3, 1),
                    "expected_delay_saved": act.get("expected_delay_reduction", 5.0),
                    "worst_case_delay_saved": round(act.get("expected_delay_reduction", 5.0) * 0.7, 1)
                } for idx, act in enumerate(actions)
            },
            "explanations": explanations
        }

        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "optimization_search_space.json"), "w", encoding="utf-8") as f:
            json.dump(search_space, f, indent=4)

        return search_space
