class QUBOBuilder:
    def __init__(self, penalty_strength=100.0):
        self.penalty_strength = penalty_strength

    def build_qubo(self, num_vars: int, linear_costs: dict, linear_penalties: dict, quadratic_penalties: dict) -> dict:
        """
        Merges linear costs and quadratic constraint penalties to construct the QUBO matrix representation.
        Returns:
            qubo_dict (dict): Map of (idx_u, idx_v) -> coefficient (float).
            result_payload (dict): Structured diagnostic format containing num_variables, qubo_matrix, etc.
        """
        qubo_matrix = {}

        # 1. Add linear coefficients on diagonal Q_{ii}
        for idx in range(num_vars):
            cost_val = linear_costs.get(idx, 0.0)
            penalty_val = linear_penalties.get(idx, 0.0)
            val = cost_val + penalty_val
            if val != 0.0:
                qubo_matrix[(idx, idx)] = val

        # 2. Add quadratic coefficients off-diagonal Q_{ij} (i < j)
        for (idx_u, idx_v), val in quadratic_penalties.items():
            if val != 0.0:
                # Ensure ordered index pair (min, max)
                pair = (min(idx_u, idx_v), max(idx_u, idx_v))
                qubo_matrix[pair] = qubo_matrix.get(pair, 0.0) + val

        # 3. Compile serializable payload for logs & diagnostics
        serializable_matrix = {f"{k[0]},{k[1]}": float(v) for k, v in qubo_matrix.items()}

        payload = {
            "num_variables": num_vars,
            "qubo_matrix": serializable_matrix,
            "objective_terms": {str(k): float(v) for k, v in linear_costs.items()},
            "constraint_penalties": {
                "linear": {str(k): float(v) for k, v in linear_penalties.items()},
                "quadratic": {f"{k[0]},{k[1]}": float(v) for k, v in quadratic_penalties.items()}
            },
            "penalty_strength": float(self.penalty_strength)
        }

        return qubo_matrix, payload
