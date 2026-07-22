import time
from ai.quantum_optimization.classical_baselines import ClassicalBaselines
from ai.quantum_optimization.qaoa_optimizer import QAOAOptimizer
from ai.quantum_optimization.hybrid_optimizer import HybridOptimizer
from ai.quantum_optimization.solution_decoder import SolutionDecoder
from ai.quantum_optimization.solution_validator import SolutionValidator

class OptimizationBenchmark:
    @staticmethod
    def run_benchmark(num_vars: int, qubo_matrix: dict, reduced_vars: dict, cost_vectors: list, constraints: dict, dependencies: dict, qaoa_reps=2, qaoa_shots=1024, seed=42) -> dict:
        """
        Runs all solvers, computes exact optimality gaps, identifies categorical winners,
        and aggregates complete solver performance benchmarks.
        """
        results = {}

        # 1. Exact Solver (Ground Truth)
        bit_exact, energy_exact, time_exact = ClassicalBaselines.solve_exact(num_vars, qubo_matrix)
        act_exact = SolutionDecoder.decode_solution(bit_exact, reduced_vars)
        val_exact = SolutionValidator.validate_plan(act_exact, constraints, dependencies)
        results["exact"] = {
            "bitstring": bit_exact,
            "energy": energy_exact,
            "runtime_seconds": time_exact,
            "actions": act_exact,
            "validation": val_exact,
            "optimality_gap_percent": 0.0
        }

        # Helper to compute optimality gap
        def calc_gap(energy):
            if abs(energy_exact) < 1e-9:
                return 0.0 if abs(energy - energy_exact) < 1e-9 else 100.0
            return round((abs(energy - energy_exact) / abs(energy_exact)) * 100.0, 2)

        # 2. Greedy Solver
        bit_greedy, energy_greedy, time_greedy = ClassicalBaselines.solve_greedy(num_vars, qubo_matrix)
        act_greedy = SolutionDecoder.decode_solution(bit_greedy, reduced_vars)
        val_greedy = SolutionValidator.validate_plan(act_greedy, constraints, dependencies)
        results["greedy"] = {
            "bitstring": bit_greedy,
            "energy": energy_greedy,
            "runtime_seconds": time_greedy,
            "actions": act_greedy,
            "validation": val_greedy,
            "optimality_gap_percent": calc_gap(energy_greedy)
        }

        # 3. Local Search Solver
        bit_ls, energy_ls, time_ls = ClassicalBaselines.solve_local_search(num_vars, qubo_matrix)
        act_ls = SolutionDecoder.decode_solution(bit_ls, reduced_vars)
        val_ls = SolutionValidator.validate_plan(act_ls, constraints, dependencies)
        results["local_search"] = {
            "bitstring": bit_ls,
            "energy": energy_ls,
            "runtime_seconds": time_ls,
            "actions": act_ls,
            "validation": val_ls,
            "optimality_gap_percent": calc_gap(energy_ls)
        }

        # 4. Simulated Annealing
        bit_sa, energy_sa, time_sa = ClassicalBaselines.solve_simulated_annealing(num_vars, qubo_matrix, seed)
        act_sa = SolutionDecoder.decode_solution(bit_sa, reduced_vars)
        val_sa = SolutionValidator.validate_plan(act_sa, constraints, dependencies)
        results["simulated_annealing"] = {
            "bitstring": bit_sa,
            "energy": energy_sa,
            "runtime_seconds": time_sa,
            "actions": act_sa,
            "validation": val_sa,
            "optimality_gap_percent": calc_gap(energy_sa)
        }

        # 5. Quantum QAOA
        qaoa_solver = QAOAOptimizer(reps=qaoa_reps, shots=qaoa_shots, seed=seed)
        qaoa_res = qaoa_solver.solve(num_vars, qubo_matrix)
        bit_qaoa = qaoa_res["bitstring"]
        energy_qaoa = qaoa_res["energy"]
        time_qaoa = qaoa_res["runtime_seconds"]
        act_qaoa = SolutionDecoder.decode_solution(bit_qaoa, reduced_vars)
        val_qaoa = SolutionValidator.validate_plan(act_qaoa, constraints, dependencies)
        results["qaoa"] = {
            "status": qaoa_res["status"],
            "backend": qaoa_res["backend"],
            "bitstring": bit_qaoa,
            "energy": energy_qaoa,
            "runtime_seconds": time_qaoa,
            "actions": act_qaoa,
            "validation": val_qaoa,
            "optimality_gap_percent": calc_gap(energy_qaoa),
            "circuit_depth": qaoa_res["circuit_depth"],
            "qubits": qaoa_res["qubits"]
        }

        # 6. Hybrid QAOA
        hybrid_res = HybridOptimizer.solve_hybrid(num_vars, qubo_matrix, qaoa_res)
        bit_hybrid = hybrid_res["refined_bitstring"]
        energy_hybrid = hybrid_res["refined_energy"]
        time_hybrid = hybrid_res["runtime_seconds"]
        act_hybrid = SolutionDecoder.decode_solution(bit_hybrid, reduced_vars)
        val_hybrid = SolutionValidator.validate_plan(act_hybrid, constraints, dependencies)
        results["hybrid_qaoa"] = {
            "status": hybrid_res["status"],
            "bitstring": bit_hybrid,
            "energy": energy_hybrid,
            "runtime_seconds": time_hybrid,
            "actions": act_hybrid,
            "validation": val_hybrid,
            "optimality_gap_percent": calc_gap(energy_hybrid)
        }

        # Helper to compute metric totals for action sets
        cost_map = {cv["action_id"]: cv["cost_vector"] for cv in cost_vectors}
        def sum_metric(actions, metric_key):
            return sum(cost_map.get(act["action_id"], {}).get(metric_key, 0.0) for act in actions)

        # 7. Identify Categorical Winners
        # Objectives to minimize: energy (cost), runtime, passenger delay, energy cost
        valid_solvers = [k for k, v in results.items() if v.get("validation", {}).get("valid", True)]
        if not valid_solvers:
            valid_solvers = ["exact", "greedy", "local_search", "simulated_annealing", "qaoa", "hybrid_qaoa"]

        # Objective Winner
        obj_winner = min(valid_solvers, key=lambda k: results[k]["energy"])
        
        # Runtime Winner
        rt_winner = min(valid_solvers, key=lambda k: results[k]["runtime_seconds"])
        
        # Delay Reduction Winner (maximizing delay reduction = minimizing delay cost)
        delay_winner = min(valid_solvers, key=lambda k: sum_metric(results[k]["actions"], "delay_saved"))
        
        # Passenger Impact Winner (minimizing passenger delay)
        pass_winner = min(valid_solvers, key=lambda k: sum_metric(results[k]["actions"], "passenger_delay"))
        
        # Energy Winner (minimizing energy consumption)
        nrg_winner = min(valid_solvers, key=lambda k: sum_metric(results[k]["actions"], "energy_consumption"))

        winners = {
            "objective_winner": obj_winner.upper(),
            "runtime_winner": rt_winner.upper(),
            "delay_reduction_winner": delay_winner.upper(),
            "passenger_impact_winner": pass_winner.upper(),
            "energy_winner": nrg_winner.upper()
        }

        return {
            "comparison": results,
            "winners": winners
        }
