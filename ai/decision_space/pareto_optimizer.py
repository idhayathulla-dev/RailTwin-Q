import json
import os

class ParetoFrontGenerator:
    @staticmethod
    def generate_pareto_front(actions: list, cost_vectors: list, passenger_impacts: dict, robustness_data: dict, data_dir="datasets") -> dict:
        """
        Determines the set of non-dominated (Pareto-optimal) solutions across multiple criteria:
        Delay reduction (minimize delay cost), Risk, Energy, Passenger Impact, Operational Cost.
        """
        candidates = []
        
        # Link all parameters together
        for idx, act in enumerate(actions):
            action_id = idx + 1
            act_type = act.get("action")
            target = act.get("train_name", act.get("station_name", "Global"))

            # Extract costs
            cost_vector = next((cv["cost_vector"] for cv in cost_vectors if cv["action_id"] == action_id), {})
            p_impact = passenger_impacts.get(str(action_id), {})
            r_data = robustness_data.get(str(action_id), {})

            candidates.append({
                "action_id": action_id,
                "action": act_type,
                "target": target,
                "delay_cost": cost_vector.get("delay", 0.0),
                "risk": cost_vector.get("risk", 0.0),
                "energy": cost_vector.get("energy", 0.0),
                "passenger_saved_percent": p_impact.get("passengers_saved", 0) / 1000.0,
                "operational_cost": cost_vector.get("operational_cost", 0.0),
                "robustness": r_data.get("robustness_score", 100) / 100.0
            })

        pareto_solutions = []
        
        # Determine non-dominated solutions
        for c1 in candidates:
            dominated = False
            for c2 in candidates:
                if c1["action_id"] == c2["action_id"]:
                    continue
                # c2 dominates c1 if c2 is at least as good in all objectives, and strictly better in at least one
                # Objectives to MINIMIZE: delay_cost, risk, energy, operational_cost
                # Objectives to MAXIMIZE: passenger_saved_percent, robustness
                c2_better_or_equal = (
                    c2["delay_cost"] <= c1["delay_cost"] and
                    c2["risk"] <= c1["risk"] and
                    c2["energy"] <= c1["energy"] and
                    c2["operational_cost"] <= c1["operational_cost"] and
                    c2["passenger_saved_percent"] >= c1["passenger_saved_percent"] and
                    c2["robustness"] >= c1["robustness"]
                )
                c2_strictly_better = (
                    c2["delay_cost"] < c1["delay_cost"] or
                    c2["risk"] < c1["risk"] or
                    c2["energy"] < c1["energy"] or
                    c2["operational_cost"] < c1["operational_cost"] or
                    c2["passenger_saved_percent"] > c1["passenger_saved_percent"] or
                    c2["robustness"] > c1["robustness"]
                )
                if c2_better_or_equal and c2_strictly_better:
                    dominated = True
                    break
            
            if not dominated:
                pareto_solutions.append(c1)

        # Fallback default if empty
        if not pareto_solutions and candidates:
            pareto_solutions = [candidates[0]]

        pareto_front = {
            "pareto_solutions": pareto_solutions,
            "total_candidates": len(candidates),
            "non_dominated_count": len(pareto_solutions)
        }

        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "pareto_front.json"), "w", encoding="utf-8") as f:
            json.dump(pareto_front, f, indent=4)

        return pareto_front
