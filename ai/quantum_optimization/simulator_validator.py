import copy
from services.movement_engine import MovementEngine
from services.state_engine import StateEngine

class SimulatorValidator:
    @staticmethod
    def run_counterfactual_simulation(network, active_events: list, current_tick: int, selected_actions: list, horizon_mins=30) -> dict:
        """
        Clones the Digital Twin, runs a forward counterfactual simulation with and without 
        the selected action plan, and measures observed delay and congestion reductions.
        """
        # 1. Run Baseline (Control: No Actions)
        baseline_network = copy.deepcopy(network)
        baseline_events = copy.deepcopy(active_events)
        
        for t in range(current_tick, current_tick + horizon_mins):
            MovementEngine.tick(baseline_network, baseline_events, t)
            StateEngine.update_occupancies(baseline_network)

        baseline_delay = sum(train.delay for train in baseline_network.trains)
        baseline_congestion = sum(s.station_congestion_score for s in baseline_network.stations) / len(baseline_network.stations) if baseline_network.stations else 0.0

        # 2. Run Optimized (Intervention: Selected Actions Applied)
        opt_network = copy.deepcopy(network)
        opt_events = copy.deepcopy(active_events)

        # Apply static modifications at start of optimization window
        for act in selected_actions:
            train_name = act.get("target", "")
            action_type = act.get("action", "")

            train_obj = opt_network.get_train_by_no(train_name)
            if not train_obj:
                # Target could be train name (e.g. 'Chennai Mail') or train number
                for t_obj in opt_network.trains:
                    if t_obj.name == train_name:
                        train_obj = t_obj
                        break
            
            if train_obj:
                if action_type == "SPEED_ADJUST":
                    train_obj.base_speed = round(train_obj.base_speed * 1.25, 1) # Speed up by 25%
                elif action_type == "PLATFORM_SWAP":
                    # Mark priority to bypass waiting for platform delays
                    train_obj.is_priority_train = True 
                elif action_type == "HOLD":
                    # Force train to hold at current position temporarily
                    train_obj.dwell_time_remaining = max(train_obj.dwell_time_remaining, 10)
                elif action_type == "REROUTE":
                    # Bypass blocking or congestion limits by reducing delay penalties
                    train_obj.base_speed = round(train_obj.base_speed * 1.1, 1)

        # Simulate forward with interventions
        for t in range(current_tick, current_tick + horizon_mins):
            # Dynamic holding or status overrides can be injected inside the loop if needed
            for act in selected_actions:
                if act.get("action") == "HOLD":
                    train_name = act.get("target", "")
                    for t_obj in opt_network.trains:
                        if t_obj.name == train_name or str(t_obj.train_no) == train_name:
                            t_obj.speed = 0.0

            MovementEngine.tick(opt_network, opt_events, t)
            StateEngine.update_occupancies(opt_network)

        opt_delay = sum(train.delay for train in opt_network.trains)
        opt_congestion = sum(s.station_congestion_score for s in opt_network.stations) / len(opt_network.stations) if opt_network.stations else 0.0

        # Calculate percentage reductions
        delay_reduction_pct = 0.0
        if baseline_delay > 0:
            delay_reduction_pct = round(((baseline_delay - opt_delay) / baseline_delay) * 100.0, 2)
            
        congestion_reduction_pct = 0.0
        if baseline_congestion > 0:
            congestion_reduction_pct = round(((baseline_congestion - opt_congestion) / baseline_congestion) * 100.0, 2)

        return {
            "baseline_delay": round(baseline_delay, 2),
            "optimized_delay": round(opt_delay, 2),
            "delay_reduction_percent": delay_reduction_pct,
            "baseline_congestion": round(baseline_congestion, 2),
            "optimized_congestion": round(opt_congestion, 2),
            "congestion_reduction_percent": congestion_reduction_pct
        }
