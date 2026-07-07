import os
import json
from ai.decision_space.cost_vector import CostVectorCalculator
from ai.decision_space.impact_estimator import ImpactEstimator
from ai.decision_space.constraint_builder import ConstraintBuilder
from ai.decision_space.action_graph import ActionDependencyGraphBuilder
from ai.decision_space.decision_validator import DecisionValidator

# Upgrade imports
from ai.decision_space.reasoning_engine import DecisionReasoningEngine
from ai.decision_space.counterfactual_engine import CounterfactualEngine
from ai.decision_space.operational_cost import OperationalCostEngine
from ai.decision_space.passenger_impact import PassengerImpactEngine
from ai.decision_space.robustness_engine import RobustnessEngine
from ai.decision_space.pareto_optimizer import ParetoFrontGenerator
from ai.decision_space.decision_embeddings import DecisionEmbeddingGenerator
from ai.decision_space.decision_explainer import DecisionExplainer
from ai.decision_space.optimization_space import OptimizationSearchSpaceGenerator
from ai.decision_space.report_generator import ReportGenerator

class DecisionBuilder:
    def __init__(self, data_dir="datasets"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.impact_json = os.path.join(self.data_dir, "decision_impact_graph.json")
        self.dep_json = os.path.join(self.data_dir, "action_dependency_graph.json")
        self.const_json = os.path.join(self.data_dir, "optimization_constraints.json")
        self.bundle_json = os.path.join(self.data_dir, "scenario_bundles.json")
        self.cost_json = os.path.join(self.data_dir, "cost_vectors.json")
        self.score_json = os.path.join(self.data_dir, "decision_scores.json")

    def build_decision_space(self, network, G, candidate_actions: list, baseline_recovery_time: int) -> dict:
        """
        Main orchestrator compiling the research-grade decision intelligence space.
        """
        # 1. Feasibility validation filter
        valid_actions = []
        for act in candidate_actions:
            if DecisionValidator.validate_action(act, network):
                valid_actions.append(act)
        
        if not valid_actions:
            valid_actions = [{"action": "SCHEDULE_MAINTENANCE", "expected_delay_reduction": 0.0, "confidence": 0.99}]

        # 2. Build prerequisites constraints
        constraints = ConstraintBuilder.build_optimization_constraints(network)

        # 3. Decision Reasoning Engine (Improvement 1)
        reasoning = DecisionReasoningEngine.generate_reasoning(valid_actions, network, self.data_dir)

        # 4. Counterfactual reasoning (Improvement 2)
        counterfactuals = CounterfactualEngine.generate_counterfactuals(valid_actions, baseline_recovery_time, self.data_dir)

        # 5. Multi-objective cost vector scaling (Improvement 3)
        cost_vectors = OperationalCostEngine.generate_costs(valid_actions, self.data_dir)

        # 6. Passenger level impact engine (Improvement 4)
        passenger_impacts = PassengerImpactEngine.calculate_impacts(valid_actions, self.data_dir)

        # 7. Robustness Analysis (Improvement 6)
        robustness = RobustnessEngine.evaluate_robustness(valid_actions, self.data_dir)

        # 8. Pareto Optimal Frontier (Improvement 8)
        pareto_front = ParetoFrontGenerator.generate_pareto_front(valid_actions, cost_vectors, passenger_impacts, robustness, self.data_dir)

        # 9. Optimization Decision Embeddings (Improvement 9)
        embeddings = DecisionEmbeddingGenerator.generate_embeddings(valid_actions, cost_vectors, passenger_impacts, robustness, self.data_dir)

        # 10. Human Explainability package (Improvement 10)
        explanations = DecisionExplainer.generate_explanations(valid_actions, cost_vectors, passenger_impacts, self.data_dir)

        # 11. Action dependency mapping (Improvement 11)
        dep_graph = ActionDependencyGraphBuilder.build_action_dependency_graph(valid_actions)

        # 12. Compile ultimate search space (Improvement 12)
        search_space = OptimizationSearchSpaceGenerator.compile_search_space(
            valid_actions, reasoning, counterfactuals, cost_vectors, passenger_impacts,
            robustness, pareto_front, embeddings, explanations, dep_graph, constraints, self.data_dir
        )

        # 13. Generate reports (Improvement 14)
        ReportGenerator.generate_all_reports(reasoning, counterfactuals, cost_vectors, robustness, pareto_front, search_space, "reports")

        # 14. Support legacy keys for compatibility
        return {
            "decision_impact_graph": reasoning,
            "action_dependency_graph": dep_graph,
            "optimization_constraints": constraints,
            "scenario_bundles": search_space.get("scenario_bundles", [
                {
                    "bundle_id": "Bundle_A",
                    "name": "Scenario A: Baseline (No Action)",
                    "actions_included": [],
                    "expected_recovery_time_mins": baseline_recovery_time,
                    "total_delay_savings_mins": 0.0,
                    "feasibility": True
                }
            ]),
            "cost_vectors": cost_vectors,
            "decision_scores": [
                {
                    "action_id": k,
                    "action": v["action"],
                    "decision_score": v["expected_improvement_mins"] * 2.0,
                    "rank": idx + 1
                } for idx, (k, v) in enumerate(reasoning.items())
            ],
            "decision_reasoning": reasoning,
            "counterfactual_analysis": counterfactuals,
            "passenger_impact": passenger_impacts,
            "robustness_report": robustness,
            "pareto_front": pareto_front,
            "decision_vectors": embeddings,
            "decision_explanations": explanations,
            "optimization_search_space": search_space
        }
