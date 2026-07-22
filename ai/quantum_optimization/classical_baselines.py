import itertools
import random
import time
import numpy as np

class ClassicalBaselines:
    @staticmethod
    def evaluate_qubo(bitstring: list, qubo_matrix: dict) -> float:
        """
        Evaluates the QUBO energy objective for a given binary bitstring.
        """
        energy = 0.0
        for (i, j), val in qubo_matrix.items():
            if i == j:
                energy += val * bitstring[i]
            else:
                energy += val * bitstring[i] * bitstring[j]
        return float(energy)

    @classmethod
    def solve_exact(cls, num_vars: int, qubo_matrix: dict) -> tuple:
        """
        Evaluates all 2^N combinations to find the exact optimal ground-truth solution.
        Optimized with NumPy bit-shifting and vectorization for speed.
        """
        start_time = time.time()
        
        if num_vars == 0:
            return [], 0.0, time.time() - start_time

        n = min(num_vars, 20)
        
        # 1. Build coefficient matrix Q
        Q = np.zeros((n, n))
        for (i, j), val in qubo_matrix.items():
            if i < n and j < n:
                Q[i, j] = val

        # 2. Vectorized generation of all states using fast bit-shifting
        num_states = 2 ** n
        x = np.zeros((num_states, n), dtype=np.int8)
        arr = np.arange(num_states, dtype=np.int32)
        for i in range(n):
            x[:, n - 1 - i] = (arr >> i) & 1

        # 3. Compute QUBO energy: energies = diag(x @ Q @ x^T) = sum((x @ Q) * x, axis=1)
        energies = np.sum((x @ Q) * x, axis=1)
        
        # 4. Find minimum index
        best_idx = np.argmin(energies)
        best_energy = float(energies[best_idx])
        best_bitstring = x[best_idx].tolist()

        # Handle fallback padding if num_vars > 20
        if len(best_bitstring) < num_vars:
            best_bitstring += [0] * (num_vars - len(best_bitstring))

        runtime = time.time() - start_time
        return best_bitstring, best_energy, runtime

    @classmethod
    def solve_greedy(cls, num_vars: int, qubo_matrix: dict) -> tuple:
        """
        Iterative greedy solver starting from all zeros.
        """
        start_time = time.time()
        current_bitstring = [0] * num_vars
        current_energy = cls.evaluate_qubo(current_bitstring, qubo_matrix)

        improved = True
        while improved:
            improved = False
            best_flip_idx = -1
            best_flip_energy = current_energy

            for idx in range(num_vars):
                test_bitstring = list(current_bitstring)
                test_bitstring[idx] = 1 - test_bitstring[idx]
                energy = cls.evaluate_qubo(test_bitstring, qubo_matrix)
                
                if energy < best_flip_energy:
                    best_flip_energy = energy
                    best_flip_idx = idx

            if best_flip_idx != -1:
                current_bitstring[best_flip_idx] = 1 - current_bitstring[best_flip_idx]
                current_energy = best_flip_energy
                improved = True

        runtime = time.time() - start_time
        return current_bitstring, current_energy, runtime

    @classmethod
    def solve_simulated_annealing(cls, num_vars: int, qubo_matrix: dict, seed=42) -> tuple:
        """
        Simulated Annealing solver with Metropolis temperature schedule.
        """
        random.seed(seed)
        start_time = time.time()
        
        current_bitstring = [random.randint(0, 1) for _ in range(num_vars)]
        current_energy = cls.evaluate_qubo(current_bitstring, qubo_matrix)
        
        best_bitstring = list(current_bitstring)
        best_energy = current_energy

        temp = 10.0
        cooling_rate = 0.98
        min_temp = 0.01

        while temp > min_temp:
            for _ in range(50):  # local steps per temp
                flip_idx = random.randint(0, num_vars - 1) if num_vars > 0 else 0
                if num_vars == 0:
                    break
                
                test_bitstring = list(current_bitstring)
                test_bitstring[flip_idx] = 1 - test_bitstring[flip_idx]
                test_energy = cls.evaluate_qubo(test_bitstring, qubo_matrix)
                
                delta = test_energy - current_energy
                if delta < 0 or random.random() < np.exp(-delta / temp):
                    current_bitstring = test_bitstring
                    current_energy = test_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_bitstring = list(current_bitstring)
            
            temp *= cooling_rate

        runtime = time.time() - start_time
        return best_bitstring, best_energy, runtime

    @classmethod
    def solve_local_search(cls, num_vars: int, qubo_matrix: dict) -> tuple:
        """
        Local Search refinement (hill climbing from a random starting point).
        """
        start_time = time.time()
        # Random initial state
        random.seed(42)
        current_bitstring = [random.randint(0, 1) for _ in range(num_vars)]
        current_energy = cls.evaluate_qubo(current_bitstring, qubo_matrix)

        improved = True
        while improved:
            improved = False
            for idx in range(num_vars):
                test_bitstring = list(current_bitstring)
                test_bitstring[idx] = 1 - test_bitstring[idx]
                test_energy = cls.evaluate_qubo(test_bitstring, qubo_matrix)
                
                if test_energy < current_energy:
                    current_bitstring = test_bitstring
                    current_energy = test_energy
                    improved = True

        runtime = time.time() - start_time
        return current_bitstring, current_energy, runtime
