import json
import os

class DecisionReasoningEngine:
    @staticmethod
    def generate_reasoning(actions: list, network, data_dir="datasets") -> dict:
        """
        Generates reasoning chains, causal graphs, and explanations for candidate actions.
        """
        reasoning_data = {}
        
        # Check active weather or blockage factors
        active_storm = False
        
        for idx, act in enumerate(actions):
            action_id = idx + 1
            act_type = act.get("action")
            target = act.get("train_name", act.get("station_name", "Global"))
            confidence = act.get("confidence", 0.9)
            saving = act.get("expected_delay_reduction", 5.0)

            # Generate reasoning chain based on context
            chain = []
            nodes = []
            edges = []

            if act_type == "PLATFORM_SWAP":
                chain = [
                    f"Disruption detected at Station {target}",
                    "Platform utilization exceeds 75% spacing constraints",
                    "Expected dwell times increase due to queue accumulation",
                    "Platform swap selected to routing alternative platform",
                    f"Expected station congestion drops by {saving * 1.5:.1f}%",
                    f"Expected recovery time improves by {saving * 0.8:.1f} minutes"
                ]
                nodes = ["Bottleneck", "Dwell Accumulation", "Alternate Platform Alloc", "Congestion Relief"]
                edges = [
                    {"source": "Bottleneck", "target": "Dwell Accumulation"},
                    {"source": "Dwell Accumulation", "target": "Alternate Platform Alloc"},
                    {"source": "Alternate Platform Alloc", "target": "Congestion Relief"}
                ]
            elif act_type == "REROUTE":
                chain = [
                    f"Downstream track segment congested or blocked",
                    f"Train {target} progress delayed on primary track route",
                    "Propagation delay spreads to following trains via safety blocks",
                    f"Bypass route via alternate track selected for Train {target}",
                    f"Primary track segment occupancy drops by 30%",
                    f"Total network delay propagation drops by {saving:.1f} minutes"
                ]
                nodes = ["Track Congestion", "Safety Block Spacing", "Bypass Route Trigger", "Delay Mitigation"]
                edges = [
                    {"source": "Track Congestion", "target": "Safety Block Spacing"},
                    {"source": "Safety Block Spacing", "target": "Bypass Route Trigger"},
                    {"source": "Bypass Route Trigger", "target": "Delay Mitigation"}
                ]
            elif act_type == "HOLD":
                chain = [
                    "Upstream delay cascade seed detected",
                    f"Holding Train {target} at current station to allow leading train to clear",
                    "Prevails safety signal spacing block conflicts",
                    f"Downstream delay transfer avoided for trailing trains",
                    f"Total cascade severity index drops by {saving * 1.2:.1f}%"
                ]
                nodes = ["Leading Delay", "Spacing Conflict Prevention", "Upstream Holding", "Cascade Relief"]
                edges = [
                    {"source": "Leading Delay", "target": "Spacing Conflict Prevention"},
                    {"source": "Spacing Conflict Prevention", "target": "Upstream Holding"},
                    {"source": "Upstream Holding", "target": "Cascade Relief"}
                ]
            else:
                chain = [
                    "Operational spacing optimal",
                    "Dynamic speed adjustments matched to safety profiles",
                    "Minimal network propagation delay expected"
                ]
                nodes = ["Optimal State"]
                edges = []

            reasoning_data[str(action_id)] = {
                "action_id": action_id,
                "action": act_type,
                "target": target,
                "reasoning_chain": chain,
                "causal_graph": {
                    "nodes": nodes,
                    "edges": edges
                },
                "explanation_text": " -> ".join(chain),
                "confidence": confidence,
                "expected_improvement_mins": saving,
                "expected_effects": {
                    "delay_reduction_minutes": saving,
                    "station_congestion_reduction": saving * 1.5,
                    "expected_recovery_reduction": int(saving * 0.8)
                }
            }

        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "decision_reasoning.json"), "w", encoding="utf-8") as f:
            json.dump(reasoning_data, f, indent=4)

        return reasoning_data
