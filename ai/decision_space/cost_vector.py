class CostVectorCalculator:
    @staticmethod
    def calculate_cost_vector(action: dict) -> dict:
        """
        Calculates normalized cost vector metrics for the candidate action.
        Positive values represent cost increases, negative values represent improvements.
        """
        act_type = action.get("action")
        
        # Base templates
        cost = {
            "delay": 0.0,
            "congestion": 0.0,
            "risk": 0.0,
            "energy": 0.0,
            "operational_cost": 0.0,
            "schedule_stability": 0.0,
            "network_resilience": 0.0
        }

        if act_type == "PLATFORM_SWAP":
            cost["delay"] = -8.5
            cost["congestion"] = -15.0
            cost["risk"] = -10.0
            cost["energy"] = 0.5
            cost["operational_cost"] = 0.8
            cost["schedule_stability"] = -2.0
            cost["network_resilience"] = -5.0
        elif act_type == "REROUTE":
            cost["delay"] = -14.2
            cost["congestion"] = -20.0
            cost["risk"] = -15.0
            cost["energy"] = 3.5
            cost["operational_cost"] = 1.8
            cost["schedule_stability"] = 8.0 # High schedule deviation cost
            cost["network_resilience"] = -12.0
        elif act_type == "HOLD":
            cost["delay"] = -6.0
            cost["congestion"] = -5.0
            cost["risk"] = -8.0
            cost["energy"] = -1.0 # Saving fuel while waiting
            cost["operational_cost"] = 0.5
            cost["schedule_stability"] = 4.0
            cost["network_resilience"] = -4.0
        elif act_type == "SPEED_ADJUST":
            cost["delay"] = -4.5
            cost["congestion"] = -8.0
            cost["risk"] = -6.0
            cost["energy"] = 2.0
            cost["operational_cost"] = 0.2
            cost["schedule_stability"] = 1.5
            cost["network_resilience"] = -3.0
            
        return cost
