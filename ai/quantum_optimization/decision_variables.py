class DecisionVariables:
    def __init__(self, max_variables=20):
        self.max_variables = max_variables

    def build_variables(self, search_space: dict) -> tuple:
        """
        Maps actions to binary decision variables x_i and reduces the search space
        to max_variables using a scalability selection pipeline.
        Returns:
            reduced_variables (dict): Map of action_id -> variable details (e.g. variable_symbol x_i).
            variable_map (dict): Map of action_id (int) -> variable index (int).
        """
        raw_variables = search_space.get("decision_variables", {})
        expected_rewards = search_space.get("expected_rewards", {})
        pareto_front = search_space.get("pareto_optimal_front", {}).get("pareto_solutions", [])
        pareto_ids = {int(p["action_id"]) for p in pareto_front}

        # 1. Feasibility Filtering
        feasible_actions = []
        for aid, details in raw_variables.items():
            action_id = int(aid)
            if details.get("feasible", True):
                # Attach reward for sorting/ranking
                reward_info = expected_rewards.get(str(action_id), {})
                delay_saved = reward_info.get("delay_reduction_mins", 0.0)
                is_pareto = action_id in pareto_ids
                feasible_actions.append({
                    "action_id": action_id,
                    "details": details,
                    "delay_saved": delay_saved,
                    "is_pareto": is_pareto
                })

        # 2. Dominated Action / Priority Ranking
        # Pareto optimal actions first, then rank by expected delay reduction
        feasible_actions = sorted(feasible_actions, key=lambda x: (x["is_pareto"], x["delay_saved"]), reverse=True)

        # 3. Top-K Action Selection (Scalability Strategy)
        selected_actions = feasible_actions[:self.max_variables]

        reduced_variables = {}
        variable_map = {}
        for idx, item in enumerate(selected_actions):
            action_id = item["action_id"]
            details = item["details"]
            symbol = f"x_{idx + 1}"
            
            reduced_variables[str(action_id)] = {
                "action_id": action_id,
                "action": details["action"],
                "target": details["target"],
                "variable_symbol": symbol,
                "index": idx
            }
            variable_map[action_id] = idx

        return reduced_variables, variable_map
