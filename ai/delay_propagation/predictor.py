import os
import csv
import json
import numpy as np
import networkx as nx
from ai.delay_propagation.graph_builder import PropagationGraphBuilder
from ai.delay_propagation.cascade_detector import CascadeDetector
from ai.delay_propagation.criticality import CriticalityEngine
from ai.delay_propagation.explainability import PropagationExplainer
from ai.delay_propagation.candidate_actions import CandidateActionGenerator

class DelayPropagationPredictor:
    def __init__(self, data_dir="datasets"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.history_csv = os.path.join(self.data_dir, "delay_propagation_history.csv")
        self.graph_json = os.path.join(self.data_dir, "propagation_graph.json")
        self.critical_json = os.path.join(self.data_dir, "critical_nodes.json")
        self.cascade_jsonl = os.path.join(self.data_dir, "cascade_events.jsonl")
        self.future_json = os.path.join(self.data_dir, "future_network_state.json")
        
        if not os.path.exists(self.history_csv):
            with open(self.history_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "tick", "cascade_severity_index", "severity_level",
                    "affected_trains_count", "affected_stations_count", "affected_tracks_count",
                    "max_propagation_depth", "expected_recovery_time_mins"
                ])

    def get_predictions_for_tick(self, network, tick: int, time_str: str, running_events: list, delay_preds: list, congestion_preds: dict) -> dict:
        """
        Main tick execution of the Research-Grade Delay Propagation Engine.
        Returns an enhanced FuturePropagationState dict ready for Layer 5 (Quantum Optimization).
        """
        # 1. Build Propagation DAG
        G = PropagationGraphBuilder.build_propagation_graph(network, running_events, delay_preds, congestion_preds)

        # 2. Run Multi-Hop Cascade Detector
        cascade_state = CascadeDetector.detect_cascading_disruptions(G)

        # 3. Calculate Criticality & Risk Scores
        train_ranks, station_ranks, track_ranks = CriticalityEngine.calculate_criticality_scores(G, network)
        risk_scores = CriticalityEngine.calculate_risk_scores(G, network)

        # 4. Generate Root Cause Trees
        disruption_explanations = PropagationExplainer.generate_all_root_causes(G)

        # 5. Deterministic CSI Calculation (Improvement 1)
        st_preds = congestion_preds.get(30, {}).get("predicted_stations", {})
        tr_preds = congestion_preds.get(30, {}).get("predicted_tracks", {})
        
        avg_st_cong = np.mean([s["congestion"] for s in st_preds.values()]) if st_preds else 0.0
        avg_tr_occ = np.mean([t["occupancy"] for t in tr_preds.values()]) if tr_preds else 0.0
        avg_tr_delay = np.mean([p["delay_predictions"]["30"] for p in delay_preds]) if delay_preds else 0.0

        csi = min(100.0, round(avg_st_cong * 0.4 + avg_tr_occ * 0.4 + avg_tr_delay * 1.5, 1))

        if csi <= 20: severity = "Normal"
        elif csi <= 40: severity = "Minor"
        elif csi <= 60: severity = "Moderate"
        elif csi <= 80: severity = "Severe"
        else: severity = "Critical"

        # 6. Deterministic Expected Recovery Time (ERT) (Improvement 1)
        c_30 = cascade_state.get(30, {"max_depth": 0})
        depth = c_30["max_depth"]
        
        active_event_durations = [ev.duration for ev in running_events if ev.active]
        remaining_disruption = max(active_event_durations) if active_event_durations else 0
        
        # If there are no disruptions or delays, expected recovery is 0
        expected_recovery_time = max(0, remaining_disruption + int(depth * 6)) if (remaining_disruption > 0 or avg_tr_delay > 5.0) else 0

        # 7. Generate Candidate Actions & Scenario Comparisons
        actions = CandidateActionGenerator.generate_candidate_actions(G, network)
        scenarios = CandidateActionGenerator.compare_scenarios(actions, expected_recovery_time)

        # 8. Build Layer 4 Pre-Quantum Decision Space (Improvement 4)
        from ai.decision_space.decision_builder import DecisionBuilder as OptDecisionBuilder
        opt_builder = OptDecisionBuilder(data_dir=self.data_dir)
        decision_space = opt_builder.build_decision_space(network, G, actions, expected_recovery_time)

        # 9. Write Logs
        with open(self.history_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                tick, csi, severity, len(c_30.get("affected_trains", [])), 
                len(c_30.get("affected_stations", [])), len(c_30.get("affected_tracks", [])), 
                depth, expected_recovery_time
            ])

        graph_data = nx.node_link_data(G)
        with open(self.graph_json, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=4)

        critical_nodes = {
            "critical_trains": train_ranks,
            "critical_stations": station_ranks,
            "critical_tracks": track_ranks
        }
        with open(self.critical_json, "w", encoding="utf-8") as f:
            json.dump(critical_nodes, f, indent=4)

        with open(self.cascade_jsonl, "a", encoding="utf-8") as f:
            for exp in disruption_explanations:
                f.write(json.dumps({"tick": tick, **exp}) + "\n")

        # Compile Enhanced FuturePropagationState (Future Network State for Layer 5 Quantum optimizer)
        global_conf = round(float(95.0 - depth * 1.5), 1)
        future_state = {
            "future_state": {
                "tick": tick,
                "time": time_str,
                "cascade_severity_index": csi,
                "severity_level": severity,
                "cascade_forecasts": cascade_state
            },
            "cascade_graph": graph_data,
            "critical_nodes": critical_nodes,
            "risk_scores": risk_scores,
            "uncertainty": {
                "confidence_score": global_conf,
                "uncertainty_score": round(1.0 - global_conf / 100.0, 3)
            },
            "candidate_actions": actions,
            "decision_impact_graph": decision_space["decision_impact_graph"],
            "action_dependency_graph": decision_space["action_dependency_graph"],
            "optimization_constraints": decision_space["optimization_constraints"],
            "scenario_bundles": decision_space["scenario_bundles"],
            "cost_vectors": decision_space["cost_vectors"],
            "decision_scores": decision_space["decision_scores"],
            "decision_reasoning": decision_space["decision_reasoning"],
            "counterfactual_analysis": decision_space["counterfactual_analysis"],
            "passenger_impact": decision_space["passenger_impact"],
            "robustness_report": decision_space["robustness_report"],
            "pareto_front": decision_space["pareto_front"],
            "decision_vectors": decision_space["decision_vectors"],
            "decision_explanations": decision_space["decision_explanations"],
            "optimization_search_space": decision_space["optimization_search_space"],
            "expected_recovery": {
                "expected_recovery_time_mins": expected_recovery_time,
                "stabilization_time": expected_recovery_time + tick,
                "confidence": global_conf
            },
            "confidence": global_conf
        }

        with open(self.future_json, "w", encoding="utf-8") as f:
            json.dump(future_state, f, indent=4)

        return future_state
