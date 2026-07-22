class ConstraintEncoder:
    def __init__(self, penalty_strength=100.0):
        self.penalty_strength = penalty_strength

    def encode_constraints(self, variable_map: dict, dependencies: dict, constraints: dict) -> tuple:
        """
        Translates conflicts, exclusions, and platform/track capacities into linear and quadratic QUBO penalties.
        Returns:
            linear_penalties (dict): idx -> linear penalty adjustment (float).
            quadratic_penalties (dict): (idx_u, idx_v) -> quadratic penalty adjustment (float).
        """
        linear_penalties = {}
        quadratic_penalties = {}

        # 1. Dependency Graph Relations (Conflicts, Prerequisites)
        edges = dependencies.get("edges", [])
        nodes = dependencies.get("nodes", {})

        # Process edges for quadratic conflict penalties
        for edge in edges:
            src_id = int(edge.get("source", 0))
            tgt_id = int(edge.get("target", 0))
            rel = edge.get("relationship", "")

            if src_id in variable_map and tgt_id in variable_map:
                idx_u = variable_map[src_id]
                idx_v = variable_map[tgt_id]
                pair = (min(idx_u, idx_v), max(idx_u, idx_v))

                if rel == "CONFLICTS_WITH" or rel == "INCOMPATIBLE":
                    # Conflict: penalty * x_u * x_v
                    quadratic_penalties[pair] = quadratic_penalties.get(pair, 0.0) + self.penalty_strength
                elif rel == "REQUIRES":
                    # Requires: target requires source (e.g., target_action requires source_action)
                    # We penalize: target = 1 and source = 0.
                    # Penalty term: penalty * x_tgt * (1 - x_src) = penalty * x_tgt - penalty * x_src * x_tgt
                    # Linear coefficient for target increases:
                    linear_penalties[idx_v] = linear_penalties.get(idx_v, 0.0) + self.penalty_strength
                    # Quadratic coefficient for target-source decreases:
                    quadratic_penalties[pair] = quadratic_penalties.get(pair, 0.0) - self.penalty_strength

        # Process nodes for internal conflicts/requirements not represented as explicit edges
        for aid_str, detail in nodes.items():
            aid = int(aid_str)
            if aid not in variable_map:
                continue
            idx_u = variable_map[aid]

            # Conflicts with
            for conf_act in detail.get("conflicts_with", []):
                # Search for variables targeting the same train/station with conf_act type
                for other_aid_str, other_detail in nodes.items():
                    other_aid = int(other_aid_str)
                    if other_aid not in variable_map or other_aid == aid:
                        continue
                    if other_detail.get("action") == conf_act and other_detail.get("train_id") == detail.get("train_id"):
                        idx_v = variable_map[other_aid]
                        pair = (min(idx_u, idx_v), max(idx_u, idx_v))
                        quadratic_penalties[pair] = quadratic_penalties.get(pair, 0.0) + self.penalty_strength

        # 2. Platform & Track Capacity Penalties
        # e.g., if multiple platform swaps require the same station platform, and available_slots is limited
        # For simplicity, if two actions target the same station, and available_slots < 2, they conflict
        platforms = constraints.get("platform_capacities", {})
        for stat_key, plat_info in platforms.items():
            avail = plat_info.get("available_slots", 1)
            if avail < 2:
                # Find all swap actions targeting this station
                station_id = int(stat_key.split("_")[-1]) if "_" in stat_key else 0
                swaps_at_station = []
                for aid, idx in variable_map.items():
                    node_detail = nodes.get(str(aid), {})
                    if node_detail.get("action") == "PLATFORM_SWAP" and node_detail.get("station_id") == station_id:
                        swaps_at_station.append(idx)
                
                # Pairwise conflict for swaps exceeding capacity
                for i in range(len(swaps_at_station)):
                    for j in range(i + 1, len(swaps_at_station)):
                        idx_u = swaps_at_station[i]
                        idx_v = swaps_at_station[j]
                        pair = (min(idx_u, idx_v), max(idx_u, idx_v))
                        quadratic_penalties[pair] = quadratic_penalties.get(pair, 0.0) + self.penalty_strength

        return linear_penalties, quadratic_penalties
