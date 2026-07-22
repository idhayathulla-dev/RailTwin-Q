import time
import numpy as np
from scipy.optimize import minimize

class QAOAOptimizer:
    def __init__(self, reps=2, shots=1024, seed=42):
        self.reps = reps
        self.shots = shots
        self.seed = seed

    def solve(self, num_vars: int, qubo_matrix: dict, method="COBYLA", n_starts=3, 
              mode="SIMULATOR", depolarizing_noise=0.0, bit_flip_noise=0.0, readout_noise=0.0) -> dict:
        """
        Executes QAOA using an advanced statevector simulator supporting:
        - Mode separation (SIMULATOR, AER, IBM_QUANTUM)
        - Circuit resource bounds & memory safeguards
        - Mathematical depolarizing, bit-flip, and readout noise emulation.
        """
        start_time = time.time()
        np.random.seed(self.seed)
        
        # 1. Mode Checks & Separations
        if mode == "IBM_QUANTUM":
            return {
                "status": "UNAVAILABLE",
                "backend": "IBM Quantum Hardware Backend",
                "reason": "IBM Quantum = NOT EXECUTED (Credentials unavailable)",
                "runtime_seconds": time.time() - start_time,
                "bitstring": [0] * num_vars,
                "energy": 0.0
            }
            
        if num_vars == 0:
            return {
                "status": "SUCCESS",
                "bitstring": [],
                "energy": 0.0,
                "qubits": 0,
                "circuit_depth": 0,
                "runtime_seconds": time.time() - start_time,
                "backend": "NumPyVectorSimulator",
                "top_k_bitstrings": [[]]
            }

        # 2. Resource safeguards & aborts
        # 2^25 complex states is 33 million floats (approx 512 MB memory)
        # depth reps * N > 100 triggers safeguard limits to prevent CPU hang
        num_pairs = len([1 for (i, j) in qubo_matrix.keys() if i != j])
        circuit_depth = self.reps * (num_vars + 2 * num_pairs)
        gate_count = self.reps * (num_vars + 3 * num_pairs)
        two_qubit_gates = self.reps * 2 * num_pairs
        
        if num_vars >= 25 or circuit_depth >= 150:
            return {
                "status": "ABORTED",
                "backend": "NumPyVectorSimulator",
                "reason": f"Resource Limit Exceeded (Qubits: {num_vars}, Depth: {circuit_depth})",
                "runtime_seconds": time.time() - start_time,
                "bitstring": [0] * num_vars,
                "energy": 0.0
            }

        # 3. Generate all state configurations
        num_states = 2 ** num_vars
        arr = np.arange(num_states, dtype=np.int32)
        x = np.zeros((num_states, num_vars), dtype=np.int8)
        for i in range(num_vars):
            x[:, num_vars - 1 - i] = (arr >> i) & 1

        # 4. Build QUBO cost coefficient array
        Q = np.zeros((num_vars, num_vars))
        for (i, j), val in qubo_matrix.items():
            if i < num_vars and j < num_vars:
                Q[i, j] = val

        costs = np.sum((x @ Q) * x, axis=1)

        # 5. Define QAOA ansatz simulation
        def get_qaoa_state(gamma, beta):
            # Start in equal superposition |+>
            state = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
            
            # Apply reps of Cost and Mixer unitaries
            for r in range(self.reps):
                # Apply Cost Unitary: exp(-i * gamma * H_C)
                state = state * np.exp(-1j * gamma * costs)
                
                # Apply Mixer Unitary: exp(-i * beta * H_M) -> Rx(2*beta) on each qubit
                cos_b = np.cos(beta)
                sin_b = -1j * np.sin(beta)
                
                for q in range(num_vars):
                    state = state.reshape((2**(num_vars - 1 - q), 2, 2**q))
                    s0 = state[:, 0, :].copy()
                    s1 = state[:, 1, :].copy()
                    state[:, 0, :] = cos_b * s0 + sin_b * s1
                    state[:, 1, :] = sin_b * s0 + cos_b * s1
                
                state = state.flatten()
            return state

        # 6. Define cost expectation value function
        def objective(params):
            gamma, beta = params
            state = get_qaoa_state(gamma, beta)
            probs = np.abs(state) ** 2
            
            # Apply noise channels to objective parameter selection
            if depolarizing_noise > 0.0:
                probs = (1.0 - depolarizing_noise) * probs + (depolarizing_noise / num_states)
            return float(np.sum(probs * costs))

        # 7. Multi-start parameter search to avoid local minima
        best_fun = float("inf")
        best_params = [0.1, 0.1]
        
        for s in range(n_starts):
            init_params = np.random.uniform(0.0, np.pi, 2).tolist()
            res = minimize(objective, init_params, method=method, options={"maxiter": 100})
            if res.fun < best_fun:
                best_fun = res.fun
                best_params = res.x

        gamma_opt, beta_opt = best_params
        
        # 8. Retrieve optimal state probabilities and bitstring
        final_state = get_qaoa_state(gamma_opt, beta_opt)
        probs = np.abs(final_state) ** 2
        
        # Apply noise sweeps on final measurements
        # Depolarizing Noise: mixes statevector with maximally mixed uniform distribution
        if depolarizing_noise > 0.0:
            probs = (1.0 - depolarizing_noise) * probs + (depolarizing_noise / num_states)
            
        # Bit-Flip Noise: flip each state configuration qubit with probability bit_flip_noise
        if bit_flip_noise > 0.0:
            noisy_probs = np.zeros(num_states)
            for idx in range(num_states):
                current_config = x[idx]
                for flip_idx in range(num_states):
                    flip_config = x[flip_idx]
                    diff_bits = np.sum(current_config != flip_config)
                    transition_prob = (bit_flip_noise ** diff_bits) * ((1.0 - bit_flip_noise) ** (num_vars - diff_bits))
                    noisy_probs[idx] += probs[flip_idx] * transition_prob
            probs = noisy_probs

        # Readout Noise: flips final readout measurement values
        if readout_noise > 0.0:
            noisy_probs = np.zeros(num_states)
            for idx in range(num_states):
                current_config = x[idx]
                for flip_idx in range(num_states):
                    flip_config = x[flip_idx]
                    diff_bits = np.sum(current_config != flip_config)
                    transition_prob = (readout_noise ** diff_bits) * ((1.0 - readout_noise) ** (num_vars - diff_bits))
                    noisy_probs[idx] += probs[flip_idx] * transition_prob
            probs = noisy_probs

        # 9. Extract outputs
        best_idx = np.argmax(probs)
        best_bitstring = x[best_idx].tolist()
        best_energy = float(costs[best_idx])

        # Extract top-K bitstrings (extract up to 8 top configurations)
        K = min(8, num_states)
        top_k_indices = np.argsort(probs)[::-1][:K]
        top_k_bitstrings = [x[idx].tolist() for idx in top_k_indices]

        runtime = time.time() - start_time

        return {
            "status": "SUCCESS",
            "bitstring": best_bitstring,
            "energy": best_energy,
            "qubits": num_vars,
            "circuit_depth": circuit_depth,
            "gate_count": gate_count,
            "two_qubit_gate_count": two_qubit_gates,
            "runtime_seconds": runtime,
            "backend": "NumPyVectorSimulator",
            "top_k_bitstrings": top_k_bitstrings,
            "optimal_parameters": best_params.tolist()
        }
