class ObjectiveFunction:
    DEFAULT_WEIGHTS = {
        "delay": 0.40,
        "congestion": 0.15,
        "passenger": 0.15,
        "risk": 0.15,
        "energy": 0.05,
        "operational_cost": 0.05,
        "schedule_stability": 0.05
    }

    def __init__(self, weights=None):
        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS
        self.structured_schemas = {}

    def calculate_linear_coefficients(self, variable_map: dict, cost_vectors: list) -> dict:
        """
        Computes the linear coefficient for each decision variable in the QUBO based on the strict multi-objective schema.
        Prevents the QUBO builder from mixing raw and normalized values.
        """
        linear_coefficients = {}
        cost_map = {cv["action_id"]: cv["cost_vector"] for cv in cost_vectors}

        for action_id, idx in variable_map.items():
            cost_vector = cost_map.get(action_id, {})
            
            # 1. Delay saved: map theoretical delay saved to 30-minute predicted delay saving
            theoretical = abs(cost_vector.get("delay", 0.0))
            predicted_30 = min(theoretical * 0.15, 5.0)  # estimated saving in 30min horizon
            
            # Map cost parameters into the strict schema
            schema = {
                "delay_reduction_minutes": {
                    "raw": theoretical,
                    "normalized": float(-predicted_30 / 10.0),  # negative cost represents benefit
                    "weight": self.weights.get("delay", 0.40),
                    "direction": "maximize",
                    "unit": "minutes",
                    "source": "Layer 4 Expected Effects"
                },
                "congestion": {
                    "raw": float(cost_vector.get("platform_usage", 0.0) * 0.5 + cost_vector.get("track_utilization", 0.0) * 0.5),
                    "normalized": float(cost_vector.get("platform_usage", 0.0) * 0.5 + cost_vector.get("track_utilization", 0.0) * 0.5),
                    "weight": self.weights.get("congestion", 0.15),
                    "direction": "minimize",
                    "unit": "occupancy_ratio",
                    "source": "Layer 3 Congestion Output"
                },
                "passenger_impact": {
                    "raw": float(cost_vector.get("passenger_delay", 0.0)),
                    "normalized": float(cost_vector.get("passenger_delay", 0.0)),
                    "weight": self.weights.get("passenger", 0.15),
                    "direction": "minimize",
                    "unit": "index",
                    "source": "Layer 4 Passenger Impact Engine"
                },
                "safety_risk": {
                    "raw": float(cost_vector.get("safety_risk", 0.0)),
                    "normalized": float(cost_vector.get("safety_risk", 0.0)),
                    "weight": self.weights.get("risk", 0.15),
                    "direction": "minimize",
                    "unit": "risk_index",
                    "source": "Layer 4 Risk Analysis"
                },
                "energy": {
                    "raw": float(cost_vector.get("energy_consumption", 0.0)),
                    "normalized": float(cost_vector.get("energy_consumption", 0.0)),
                    "weight": self.weights.get("energy", 0.05),
                    "direction": "minimize",
                    "unit": "kWh",
                    "source": "Layer 4 Operational Cost"
                },
                "operational_cost": {
                    "raw": float(cost_vector.get("operational_complexity", 0.0)),
                    "normalized": float(cost_vector.get("operational_complexity", 0.0)),
                    "weight": self.weights.get("operational_cost", 0.05),
                    "direction": "minimize",
                    "unit": "index",
                    "source": "Layer 4 Cost Engine"
                },
                "schedule_stability": {
                    "raw": float(cost_vector.get("schedule_stability", 0.0)),
                    "normalized": float(cost_vector.get("schedule_stability", 0.0)),
                    "weight": self.weights.get("schedule_stability", 0.05),
                    "direction": "minimize",
                    "unit": "index",
                    "source": "Layer 4 Robustness"
                }
            }

            self.structured_schemas[action_id] = schema

            # 2. Compute strictly weighted normalized coefficients
            cost_value = sum(
                meta["weight"] * meta["normalized"] for meta in schema.values()
            )
            linear_coefficients[idx] = float(cost_value)

        return linear_coefficients
