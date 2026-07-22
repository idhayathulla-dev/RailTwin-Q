class SolutionDecoder:
    @staticmethod
    def decode_solution(bitstring: list, reduced_variables: dict) -> list:
        """
        Decodes the binary bitstring back into a list of selected operational actions.
        """
        selected_actions = []
        
        # Invert reduced_variables mapping to index -> action details
        idx_map = {details["index"]: details for details in reduced_variables.values()}

        for idx, val in enumerate(bitstring):
            if val == 1 and idx in idx_map:
                details = idx_map[idx]
                selected_actions.append({
                    "action_id": details["action_id"],
                    "action": details["action"],
                    "target": details["target"]
                })

        return selected_actions
