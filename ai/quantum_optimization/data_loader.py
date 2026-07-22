import json
import os

class OptimizationDataLoader:
    def __init__(self, data_dir="datasets"):
        self.data_dir = data_dir
        self.search_space_path = os.path.join(self.data_dir, "optimization_search_space.json")

    def load_search_space(self) -> dict:
        """
        Loads the pre-quantum optimization search space compiled by Layer 4.
        """
        if not os.path.exists(self.search_space_path):
            raise FileNotFoundError(f"Optimization search space file not found: {self.search_space_path}")

        with open(self.search_space_path, "r", encoding="utf-8") as f:
            search_space = json.load(f)
            
        return {
            "decision_variables": search_space.get("decision_variables", {}),
            "constraints": search_space.get("constraints", {}),
            "cost_vectors": search_space.get("cost_vectors", []),
            "expected_rewards": search_space.get("expected_rewards", {}),
            "dependencies": search_space.get("dependencies", {}),
            "pareto_optimal_front": search_space.get("pareto_optimal_front", {}),
            "decision_embeddings": search_space.get("decision_embeddings", {}),
            "confidence_intervals": search_space.get("confidence_intervals", {}),
            "explanations": search_space.get("explanations", {})
        }
