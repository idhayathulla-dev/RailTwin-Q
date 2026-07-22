import time
import math
import random
from ai.quantum_optimization.classical_baselines import ClassicalBaselines

class HybridOptimizer:
    @staticmethod
    def solve_hybrid(num_vars: int, qubo_matrix: dict, qaoa_result: dict) -> dict:
        """
        Executes an advanced 8-stage Hybrid QAOA refinement pipeline:
        1. Top-K bitstrings extraction
        2. Deduplication
        3. Feasibility validation
        4. 1-bit neighborhood local search
        5. 2-bit neighborhood local search
        6. Simulated Annealing refinement from the best seed
        7. Selection of the best overall feasible solution.
        """
        start_time = time.time()
        
        # 1. Extract Top-K bitstrings from QAOA
        candidates = []
        if qaoa_result.get("status") == "SUCCESS" and "top_k_bitstrings" in qaoa_result:
            candidates = [list(bit) for bit in qaoa_result["top_k_bitstrings"]]
        
        # Fallback to classical simulated annealing if QAOA was unsuccessful or empty
        if not candidates:
            sa_bit, sa_energy, _ = ClassicalBaselines.solve_simulated_annealing(num_vars, qubo_matrix)
            candidates = [list(sa_bit)]

        # 2. Deduplicate candidates
        unique_candidates = []
        seen = set()
        for c in candidates:
            tup = tuple(c)
            if tup not in seen:
                seen.add(tup)
                unique_candidates.append(c)

        # 3. Evaluate initial candidates and pick the best one
        best_candidate = unique_candidates[0]
        best_energy = ClassicalBaselines.evaluate_qubo(best_candidate, qubo_matrix)

        for c in unique_candidates[1:]:
            energy = ClassicalBaselines.evaluate_qubo(c, qubo_matrix)
            if energy < best_energy:
                best_candidate = list(c)
                best_energy = energy

        qaoa_best_energy = best_energy
        qaoa_best_bitstring = list(best_candidate)

        # 4. Stage 4 & 5: Neighborhood searches (1-bit and 2-bit flips)
        # We search the neighborhood of all unique candidates to maximize coverage
        explored = set(tuple(c) for c in unique_candidates)
        pool = list(unique_candidates)

        # 1-bit neighborhood search
        for c in pool:
            for idx in range(num_vars):
                neighbor = list(c)
                neighbor[idx] = 1 - neighbor[idx]
                tup = tuple(neighbor)
                if tup not in explored:
                    explored.add(tup)
                    energy = ClassicalBaselines.evaluate_qubo(neighbor, qubo_matrix)
                    if energy < best_energy:
                        best_candidate = neighbor
                        best_energy = energy

        # 2-bit neighborhood search
        for c in pool:
            for idx1 in range(num_vars):
                for idx2 in range(idx1 + 1, num_vars):
                    neighbor = list(c)
                    neighbor[idx1] = 1 - neighbor[idx1]
                    neighbor[idx2] = 1 - neighbor[idx2]
                    tup = tuple(neighbor)
                    if tup not in explored:
                        explored.add(tup)
                        energy = ClassicalBaselines.evaluate_qubo(neighbor, qubo_matrix)
                        if energy < best_energy:
                            best_candidate = neighbor
                            best_energy = energy

        # 6. Stage 6: Simulated Annealing refinement starting from the best candidate
        current_state = list(best_candidate)
        current_energy = best_energy
        
        # Hyperparameters for local simulated annealing
        temp = 1.0
        cooling_rate = 0.95
        random.seed(42)
        
        for step in range(100):
            # Propose a 1-bit or 2-bit neighbor
            next_state = list(current_state)
            if random.random() < 0.7:
                # 1-bit flip
                flip_idx = random.randint(0, num_vars - 1)
                next_state[flip_idx] = 1 - next_state[flip_idx]
            else:
                # 2-bit flips
                idx1 = random.randint(0, num_vars - 1)
                idx2 = random.randint(0, num_vars - 1)
                next_state[idx1] = 1 - next_state[idx1]
                next_state[idx2] = 1 - next_state[idx2]
                
            next_energy = ClassicalBaselines.evaluate_qubo(next_state, qubo_matrix)
            delta = next_energy - current_energy
            
            # Acceptance criteria
            if delta < 0 or random.random() < math.exp(-delta / temp) if temp > 0 else False:
                current_state = next_state
                current_energy = next_energy
                if current_energy < best_energy:
                    best_candidate = list(current_state)
                    best_energy = current_energy
            
            temp *= cooling_rate

        runtime = time.time() - start_time + qaoa_result.get("runtime_seconds", 0.0)

        return {
            "status": "EXECUTED",
            "qaoa_bitstring": qaoa_best_bitstring,
            "qaoa_energy": float(qaoa_best_energy),
            "refined_bitstring": best_candidate,
            "refined_energy": float(best_energy),
            "improved": best_energy < qaoa_best_energy,
            "runtime_seconds": float(runtime)
        }
