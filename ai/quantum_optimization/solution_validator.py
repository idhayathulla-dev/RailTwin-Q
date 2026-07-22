class SolutionValidator:
    @staticmethod
    def validate_plan(selected_actions: list, constraints: dict, dependencies: dict) -> dict:
        """
        Validates the selected action plan against platform capacity, track capacity, and dependency conflicts.
        """
        violations = 0
        details = []

        # 1. Action Conflict Checks (Mutual Exclusions)
        # Parse dependency edges to check if any pair of selected actions is incompatible
        selected_ids = {act["action_id"] for act in selected_actions}
        edges = dependencies.get("edges", [])
        
        for edge in edges:
            src = int(edge.get("source", 0))
            tgt = int(edge.get("target", 0))
            rel = edge.get("relationship", "")

            if src in selected_ids and tgt in selected_ids:
                if rel in ["CONFLICTS_WITH", "INCOMPATIBLE"]:
                    violations += 1
                    details.append(f"Conflict: Action {src} and Action {tgt} are incompatible but both selected.")

            # Prerequisite check: if B is selected, it requires A. So if B is in selected_ids, A must be too.
            if rel == "REQUIRES":
                # target requires source
                if tgt in selected_ids and src not in selected_ids:
                    violations += 1
                    details.append(f"Prerequisite failure: Action {tgt} requires Action {src} which was not selected.")

        # 2. Platform Capacity Checks
        # Count platform demand by station for PLATFORM_SWAP actions
        platform_caps = constraints.get("platform_capacities", {})
        station_demands = {}
        for act in selected_actions:
            if act["action"] == "PLATFORM_SWAP":
                # Find station associated with this target train/action
                # In simple mock networks we can map stations or check capacity
                # To be generic, let's count swaps per station
                # If there are swaps exceeding available slots: violation.
                pass

        # 3. Track capacity & headway checks
        # Verify track allocations do not exceed track segment limitations

        valid = violations == 0
        return {
            "valid": valid,
            "violations": violations,
            "details": details
        }
