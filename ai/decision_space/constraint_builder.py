class ConstraintBuilder:
    @staticmethod
    def build_optimization_constraints(network) -> dict:
        """
        Compiles structural and operational constraints from the Digital Twin configuration.
        """
        platform_limits = {}
        for s in network.stations:
            platform_limits[f"station_{s.station_id}"] = {
                "max_platforms": s.platforms,
                "current_platforms_occupied": s.platforms_occupied,
                "available_slots": max(0, s.platforms - s.platforms_occupied)
            }

        track_limits = {}
        for tr in network.tracks:
            track_limits[f"track_{tr.track_id}"] = {
                "capacity": tr.capacity,
                "current_trains": tr.current_trains,
                "blocked": tr.blocked
            }

        return {
            "platform_capacities": platform_limits,
            "track_capacities": track_limits,
            "safety_rules": {
                "headway_spacing_mins": 5,
                "express_train_precedence": True,
                "priority_preemption": True
            },
            "operating_limits": {
                "crew_rest_mins_simplified": 30,
                "maintenance_blocks_active": [tr.track_id for tr in network.tracks if tr.blocked]
            }
        }
