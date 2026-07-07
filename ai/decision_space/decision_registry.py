import os
import json
import datetime

class DecisionRegistry:
    def __init__(self, registry_dir="models/congestion_predictor"):
        self.registry_path = os.path.join(registry_dir, "model_registry.json")
        os.makedirs(registry_dir, exist_ok=True)
        
    def register_decision_engine(self, version: str = "v1.0.0"):
        """
        Saves metadata version records for Layer 4 Decision Intelligence Engine.
        """
        registry_data = {}
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    registry_data = json.load(f)
            except Exception:
                pass
                
        registry_data["decision_intelligence_engine"] = {
            "version": version,
            "registered_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Production-Ready",
            "capabilities": [
                "decision_impact_graph",
                "action_dependency_graph",
                "optimization_constraints",
                "scenario_bundles",
                "cost_vectors",
                "decision_scores"
            ]
        }
        
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=4)
        return True
# Auto-register
DecisionRegistry().register_decision_engine()
