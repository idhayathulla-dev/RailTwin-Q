import os
import json
import time
from ai.quantum_optimization.data_loader import OptimizationDataLoader
from ai.quantum_optimization.decision_variables import DecisionVariables
from ai.quantum_optimization.objective_function import ObjectiveFunction
from ai.quantum_optimization.constraint_encoder import ConstraintEncoder
from ai.quantum_optimization.qubo_builder import QUBOBuilder
from ai.quantum_optimization.benchmark import OptimizationBenchmark
from ai.quantum_optimization.solution_decoder import SolutionDecoder
from ai.quantum_optimization.solution_validator import SolutionValidator
from ai.quantum_optimization.simulator_validator import SimulatorValidator
from ai.quantum_optimization.explainability import OptimizationExplainer

class QuantumOrchestrator:
    def __init__(self, data_dir="datasets", reports_dir="reports", reps=2, shots=1024, seed=42):
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        self.reps = reps
        self.shots = shots
        self.seed = seed
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def optimize_network(self, network, active_events: list, tick: int, baseline_recovery_time: int) -> dict:
        """
        Coordinates the entire Layer 5 optimization, validation, and benchmarking pipeline.
        Writes logs to datasets/optimization_result.json and reports/quantum_benchmark_report.html.
        """
        start_time = time.time()
        
        # 1. Load data
        loader = OptimizationDataLoader(data_dir=self.data_dir)
        search_space = loader.load_search_space()

        # 2. Variable reduction & mapping
        var_engine = DecisionVariables(max_variables=10)
        reduced_vars, variable_map = var_engine.build_variables(search_space)
        num_vars = len(reduced_vars)

        if num_vars == 0:
            # Empty decision space scenario
            return self._build_empty_payload(tick, baseline_recovery_time)

        # 3. Objective coefficients
        obj_engine = ObjectiveFunction()
        linear_costs = obj_engine.calculate_linear_coefficients(variable_map, search_space["cost_vectors"])

        # 4. Constraint encoder
        const_engine = ConstraintEncoder(penalty_strength=100.0)
        linear_penalties, quadratic_penalties = const_engine.encode_constraints(
            variable_map, search_space["dependencies"], search_space["constraints"]
        )

        # 5. Build QUBO
        qubo_engine = QUBOBuilder(penalty_strength=100.0)
        qubo_matrix, qubo_payload = qubo_engine.build_qubo(
            num_vars, linear_costs, linear_penalties, quadratic_penalties
        )

        # 6. Run Benchmark (Exact, Greedy, Simulated Annealing, Local Search, QAOA, Hybrid)
        bench = OptimizationBenchmark.run_benchmark(
            num_vars, qubo_matrix, reduced_vars, search_space["cost_vectors"],
            search_space["constraints"], search_space["dependencies"],
            qaoa_reps=self.reps, qaoa_shots=self.shots, seed=self.seed
        )

        # 7. Select best solver solution (prefer Hybrid QAOA if valid, otherwise simulated annealing)
        solver_comparison = bench["comparison"]
        selected_solver = "hybrid_qaoa"
        best_sol = solver_comparison[selected_solver]
        
        if not best_sol["validation"]["valid"]:
            selected_solver = "simulated_annealing"
            best_sol = solver_comparison[selected_solver]

        selected_actions = best_sol["actions"]
        validation_status = best_sol["validation"]

        # 8. Counterfactual Re-simulation loop
        counterfactuals = SimulatorValidator.run_counterfactual_simulation(
            network, active_events, tick, selected_actions, horizon_mins=30
        )

        # 9. Explainability
        explanations = OptimizationExplainer.generate_explanations(
            selected_actions, selected_solver.upper(), validation_status, counterfactuals
        )

        # 10. Extract Qubit metrics
        qaoa_stats = solver_comparison["qaoa"]
        qubits_used = qaoa_stats.get("qubits", 0)
        depth_used = qaoa_stats.get("circuit_depth", 0)

        # Compile final Layer 5 Result Payload
        payload = {
            "optimization_tick": tick,
            "solver_comparison": {
                k: {
                    "energy": v["energy"],
                    "runtime_seconds": round(v["runtime_seconds"], 4),
                    "optimality_gap_percent": v["optimality_gap_percent"],
                    "valid": v["validation"]["valid"],
                    "violation_count": v["validation"]["violations"]
                } for k, v in solver_comparison.items()
            },
            "selected_solver": selected_solver.upper(),
            "selected_actions": selected_actions,
            "constraint_validation": {
                "valid": validation_status["valid"],
                "violations": validation_status["violations"],
                "details": validation_status["details"]
            },
            "counterfactual_results": counterfactuals,
            "quantum_metrics": {
                "status": qaoa_stats.get("status", "UNAVAILABLE"),
                "backend": qaoa_stats.get("backend", "None"),
                "qubits": qubits_used,
                "circuit_depth": depth_used,
                "qaoa_reps": self.reps,
                "shots": self.shots
            },
            "winners": bench["winners"],
            "explanations": explanations
        }

        # Write result payload to file
        res_path = os.path.join(self.data_dir, "optimization_result.json")
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        # Generate HTML report
        self.generate_html_report(payload, solver_comparison, bench["winners"])

        return payload

    def generate_html_report(self, payload: dict, solver_comparison: dict, winners: dict):
        """
        Generates a premium dark-themed HTML report comparing solver metrics.
        """
        rows = ""
        for name, data in solver_comparison.items():
            gap = data.get("optimality_gap_percent", 0.0)
            valid = "PASS" if data["validation"]["valid"] else "FAIL"
            color = "#10b981" if valid == "PASS" else "#ef4444"
            rows += f"""
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); font-weight: 600;">{name.upper().replace('_', ' ')}</td>
                <td style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); text-align: center;">{data['energy']:.4f}</td>
                <td style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); text-align: center;">{gap:.2f}%</td>
                <td style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); text-align: center; color: {color}; font-weight: bold;">{valid} ({data['validation']['violations']} viol)</td>
                <td style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); text-align: center;">{data['runtime_seconds'] * 1000.0:.2f} ms</td>
            </tr>
            """

        actions_html = ""
        for act in payload["selected_actions"]:
            actions_html += f"<li><span style='color: var(--accent-indigo); font-weight: bold;'>[{act['action']}]</span> Target: {act['target']}</li>"
        if not actions_html:
            actions_html = "<li>No active intervention required.</li>"

        cf = payload["counterfactual_results"]

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Quantum Optimization Solver Benchmark</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --border-color: #1f2937;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-indigo: #6366f1;
            --accent-yellow: #f59e0b;
        }}
        body {{
            background: var(--bg-color);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
        }}
        h1, h2, h3 {{
            margin-top: 0;
        }}
        h1 {{ color: var(--accent-indigo); }}
        h2 {{ color: var(--text-main); border-bottom: 1px solid var(--border-color); padding-bottom: 8px; font-size: 1.3rem; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th {{
            text-align: left;
            padding: 12px;
            background: rgba(255,255,255,0.02);
            border-bottom: 1px solid var(--border-color);
            color: var(--text-muted);
            font-size: 0.85rem;
            text-transform: uppercase;
        }}
        .winner-tag {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            background: rgba(99, 102, 241, 0.15);
            color: var(--accent-indigo);
            font-weight: bold;
            font-size: 0.8rem;
        }}
        ul {{
            padding-left: 20px;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Layer 5: Quantum vs Classical Benchmark</h1>
            <p style="color: var(--text-muted); margin: 5px 0 0 0;">Optimization Tick: {payload['optimization_tick']} | Seed: {self.seed}</p>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Counterfactual Simulation Results</h2>
                <div style="display: flex; justify-content: space-around; text-align: center; margin-top: 20px;">
                    <div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--accent-red);">{cf['baseline_delay']:.1f}m</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">Baseline Delay</div>
                    </div>
                    <div style="border-left: 1px solid var(--border-color);"></div>
                    <div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--accent-green);">{cf['optimized_delay']:.1f}m</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">Optimized Delay</div>
                        <div style="color: var(--accent-green); font-size: 0.8rem; font-weight: 600; margin-top: 4px;">-{cf['delay_reduction_percent']}%</div>
                    </div>
                </div>

                <div style="display: flex; justify-content: space-around; text-align: center; margin-top: 30px; border-top: 1px solid var(--border-color); padding-top: 20px;">
                    <div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--accent-red);">{cf['baseline_congestion']:.1f}%</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">Baseline Congestion</div>
                    </div>
                    <div style="border-left: 1px solid var(--border-color);"></div>
                    <div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--accent-green);">{cf['optimized_congestion']:.1f}%</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">Optimized Congestion</div>
                        <div style="color: var(--accent-green); font-size: 0.8rem; font-weight: 600; margin-top: 4px;">-{cf['congestion_reduction_percent']}%</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Quantum Execution Configuration</h2>
                <p><strong>QAOA Status:</strong> <span style="color: var(--accent-green);">{payload['quantum_metrics']['status']}</span></p>
                <p><strong>Simulator Backend:</strong> {payload['quantum_metrics']['backend']}</p>
                <p><strong>Qubits Count:</strong> {payload['quantum_metrics']['qubits']}</p>
                <p><strong>Ising Circuit Depth:</strong> {payload['quantum_metrics']['circuit_depth']}</p>
                <p><strong>Optimization Solver:</strong> {payload['selected_solver']}</p>
                
                <h3 style="margin-top: 20px; font-size: 1rem; border-bottom: 1px solid var(--border-color); padding-bottom: 4px;">Selected Intervention Plan</h3>
                <ul style="margin: 8px 0 0 0; font-size: 0.9rem;">
                    {actions_html}
                </ul>
            </div>
        </div>

        <div class="card" style="margin-bottom: 30px;">
            <h2>Categorical Metric Winners</h2>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
                <div style="background: rgba(255,255,255,0.02); padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px;">Objective Value</div>
                    <span class="winner-tag">{winners['objective_winner']}</span>
                </div>
                <div style="background: rgba(255,255,255,0.02); padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px;">Runtime Performance</div>
                    <span class="winner-tag">{winners['runtime_winner']}</span>
                </div>
                <div style="background: rgba(255,255,255,0.02); padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px;">Delay Reduction</div>
                    <span class="winner-tag">{winners['delay_reduction_winner']}</span>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Solver Optimization Performance Matrix</h2>
            <table>
                <thead>
                    <tr>
                        <th style="text-align: left;">Optimizer Solver</th>
                        <th style="text-align: center;">Objective Value (QUBO Energy)</th>
                        <th style="text-align: center;">Optimality Gap</th>
                        <th style="text-align: center;">Constraint Feasibility</th>
                        <th style="text-align: center;">Runtime</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        rep_path = os.path.join(self.reports_dir, "quantum_benchmark_report.html")
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _build_empty_payload(self, tick: int, baseline_recovery_time: int) -> dict:
        """
        Empty fallback payload when no variables are present.
        """
        return {
            "optimization_tick": tick,
            "solver_comparison": {},
            "selected_solver": "NONE",
            "selected_actions": [],
            "constraint_validation": {"valid": True, "violations": 0, "details": []},
            "counterfactual_results": {
                "baseline_delay": 0.0,
                "optimized_delay": 0.0,
                "delay_reduction_percent": 0.0,
                "baseline_congestion": 0.0,
                "optimized_congestion": 0.0,
                "congestion_reduction_percent": 0.0
            },
            "quantum_metrics": {
                "status": "UNAVAILABLE",
                "backend": "None",
                "qubits": 0,
                "circuit_depth": 0,
                "qaoa_reps": self.reps,
                "shots": self.shots
            },
            "winners": {},
            "explanations": [{"type": "BASELINE", "text": "No candidate actions found."}]
        }
