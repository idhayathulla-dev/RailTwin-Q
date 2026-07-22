class OptimizationExplainer:
    @staticmethod
    def generate_explanations(selected_actions: list, solver_name: str, validation: dict, counterfactuals: dict) -> list:
        """
        Generates structured explanation summaries of the selected action plan, constraints, and improvements.
        """
        explanations = []

        if not selected_actions:
            explanations.append({
                "type": "BASELINE",
                "text": "No active interventions selected. Simulator continues with default timetables."
            })
            return explanations

        # Header card
        explanations.append({
            "type": "SUMMARY",
            "text": f"Optimizer {solver_name} selected {len(selected_actions)} cooperative action(s) yielding {counterfactuals.get('delay_reduction_percent', 0.0)}% delay reduction and {counterfactuals.get('congestion_reduction_percent', 0.0)}% platform/track congestion reduction."
        })

        # Actions breakdown
        for idx, act in enumerate(selected_actions):
            action_type = act.get("action")
            target = act.get("target")
            
            if action_type == "SPEED_ADJUST":
                text = f"Action {idx+1}: Speed Adjust for Train {target}. Accelerates train speed profile by 20% to clear track sections and prevent cascading spacing delays."
            elif action_type == "PLATFORM_SWAP":
                text = f"Action {idx+1}: Platform Swap at Station platform. Moves Train {target} to alternate platform, avoiding station platform gridlock delays."
            elif action_type == "HOLD":
                text = f"Action {idx+1}: Hold Train {target} at station. Intentionally delays departure to clear downstream track segment occupancy for priority trains."
            elif action_type == "REROUTE":
                text = f"Action {idx+1}: Reroute Train {target} via alternate track loop to bypass blocked or congested primary route."
            else:
                text = f"Action {idx+1}: Selected {action_type} for target {target}."
            
            explanations.append({
                "type": "ACTION_EXPLANATION",
                "action_id": act.get("action_id"),
                "text": text
            })

        # Feasibility check
        if validation.get("valid", True):
            explanations.append({
                "type": "FEASIBILITY",
                "text": "Constraint Validator check: PASS. All track/platform capacity constraints are satisfied."
            })
        else:
            explanations.append({
                "type": "FEASIBILITY",
                "text": f"Constraint Validator warning: solver plan resulted in {validation.get('violations')} constraint violations: {', '.join(validation.get('details', []))}."
            })

        return explanations
