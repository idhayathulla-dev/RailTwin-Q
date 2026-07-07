class ActionDependencyGraphBuilder:
    @staticmethod
    def build_action_dependency_graph(actions: list) -> dict:
        """
        Builds a directed dependency and compatibility graph mapping relationships between operational actions.
        """
        nodes = {}
        edges = []

        for idx, act in enumerate(actions):
            action_id = idx + 1
            act_type = act.get("action")
            
            # Formulate prerequisites, conflicts, and compatibility
            requires = []
            conflicts_with = []
            compatible_with = []

            if act_type == "PLATFORM_SWAP":
                requires = ["platform_available"]
                conflicts_with = ["HOLD"] # Cannot swap platforms while holding upstream
                compatible_with = ["SPEED_ADJUST"]
            elif act_type == "REROUTE":
                requires = ["alternative_path_exists"]
                conflicts_with = ["HOLD"]
                compatible_with = ["SPEED_ADJUST", "PLATFORM_SWAP"]
            elif act_type == "HOLD":
                requires = ["upstream_block_free"]
                conflicts_with = ["REROUTE", "SPEED_ADJUST"]
                compatible_with = ["PLATFORM_SWAP"]
            elif act_type == "SPEED_ADJUST":
                requires = ["speed_limits_within_bounds"]
                conflicts_with = ["HOLD"]
                compatible_with = ["REROUTE", "PLATFORM_SWAP"]

            nodes[str(action_id)] = {
                "action_id": action_id,
                "action": act_type,
                "requires": requires,
                "conflicts_with": conflicts_with,
                "compatible_with": compatible_with,
                "train_id": act.get("train_id", 0),
                "station_id": act.get("station_id", 0)
            }

        # Build conflict edges
        for u_id, u_attrs in nodes.items():
            for v_id, v_attrs in nodes.items():
                if u_id == v_id:
                    continue
                # If they target the same train and conflict
                if u_attrs["train_id"] == v_attrs["train_id"] and u_attrs["train_id"] > 0:
                    if v_attrs["action"] in u_attrs["conflicts_with"]:
                        edges.append({
                            "source": u_id,
                            "target": v_id,
                            "relationship": "CONFLICTS_WITH"
                        })
                # If they are compatible
                elif v_attrs["action"] in u_attrs["compatible_with"]:
                    edges.append({
                        "source": u_id,
                        "target": v_id,
                        "relationship": "COMPATIBLE"
                    })

        return {
            "nodes": nodes,
            "edges": edges
        }
