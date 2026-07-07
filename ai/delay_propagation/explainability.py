import networkx as nx

class PropagationExplainer:
    @staticmethod
    def get_causal_chain(G: nx.DiGraph, target_node: str, visited=None) -> list:
        """
        Recursively traces parent nodes back to the seed disruption source.
        Returns a list of nodes forming the causal chain.
        """
        if visited is None:
            visited = set()
            
        if target_node in visited:
            return [] # Avoid cycles in graph traversal
        visited.add(target_node)

        in_edges = G.in_edges(target_node, data=True)
        if not in_edges:
            return [target_node]
            
        # Sort by expected delay transfer to find the strongest causal contributor
        sorted_parents = sorted(in_edges, key=lambda e: e[2].get("expected_delay_transfer", 0.0), reverse=True)
        primary_parent = sorted_parents[0][0]
        
        return PropagationExplainer.get_causal_chain(G, primary_parent, visited) + [target_node]

    @staticmethod
    def explain_node_disruption(G: nx.DiGraph, target_node: str) -> dict:
        """
        Computes a complete root cause path chain for the target disrupted entity.
        """
        if target_node not in G:
            return {"node": target_node, "status": "Stable", "root_cause": "No active disruptions detected.", "causal_tree": []}

        # Traces root causal chain
        chain = PropagationExplainer.get_causal_chain(G, target_node)
        
        # Build pretty causal tree string representation
        chain_names = []
        for node in chain:
            attrs = G.nodes[node]
            n_type = attrs.get("type", "unknown")
            label = attrs.get("name", node)
            if n_type == "train":
                chain_names.append(f"Train {label} (Delay: {attrs.get('current_delay', 0.0):.1f}m)")
            elif n_type == "station":
                chain_names.append(f"Station {label} (Congestion: {attrs.get('current_congestion', 0.0):.0f}%)")
            elif n_type == "track":
                chain_names.append(f"Track {label} (Occupancy: {attrs.get('current_occupancy', 0.0):.0f}%)")
            else:
                chain_names.append(node)
                
        tree_str = " -> ".join(chain_names)

        # Get parent edge attributes for immediate cause reason
        in_edges = G.in_edges(target_node, data=True)
        if in_edges:
            sorted_parents = sorted(in_edges, key=lambda e: e[2].get("expected_delay_transfer", 0.0), reverse=True)
            primary_parent, _, edge_attrs = sorted_parents[0]
            reason = edge_attrs.get("reason", "Resource Dependency")
            p_name = G.nodes[primary_parent].get("name", primary_parent)
            root_cause = f"Congestion cascade from {p_name} via {reason}. Causal path: {tree_str}"
        else:
            root_cause = f"Primary disruption seed. Source path: {tree_str}"

        return {
            "node": target_node,
            "status": "Affected by Disruption",
            "causal_chain": chain,
            "causal_tree_path": tree_str,
            "root_cause": root_cause
        }

    @staticmethod
    def generate_all_root_causes(G: nx.DiGraph) -> list:
        """
        Generates causal path explanations for all disrupted entities.
        """
        explanations = []
        for node, attrs in G.nodes(data=True):
            n_type = attrs.get("type")
            if (n_type == "train" and attrs.get("current_delay", 0.0) > 5.0) or \
               (n_type == "station" and attrs.get("current_congestion", 0.0) > 30.0) or \
               (n_type == "track" and attrs.get("current_occupancy", 0.0) > 50.0):
                explanations.append(PropagationExplainer.explain_node_disruption(G, node))
        return explanations
