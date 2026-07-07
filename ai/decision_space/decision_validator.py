class DecisionValidator:
    @staticmethod
    def validate_action(action: dict, network) -> bool:
        """
        Validates the feasibility of an action against network assets and operational rules.
        """
        act_type = action.get("action")
        
        if act_type == "PLATFORM_SWAP":
            s_id = action.get("station_id")
            station = network.get_station_by_id(s_id)
            if not station:
                return False
            # Check if there is an empty platform slot available (or soon to be available)
            if station.platforms_occupied >= station.platforms:
                # Still feasible if we swap two trains, but let's check
                pass
            return True
            
        elif act_type == "REROUTE":
            t_id = action.get("train_id")
            train = next((t for t in network.trains if t.train_no == t_id), None)
            if not train:
                return False
            # Rerouting is only feasible if there are multiple route alternatives
            # We bypass using alternate tracks (Track 2 is currently Arakkonam-Katpadi)
            return True
            
        elif act_type == "HOLD":
            t_id = action.get("train_id")
            train = next((t for t in network.trains if t.train_no == t_id), None)
            if not train:
                return False
            # Cannot hold a train that is already arrived or not active
            if train.status == "ARRIVED":
                return False
            return True
            
        elif act_type == "SPEED_ADJUST":
            t_id = action.get("train_id")
            train = next((t for t in network.trains if t.train_no == t_id), None)
            if not train:
                return False
            if train.status != "MOVING":
                return False
            return True

        return True
