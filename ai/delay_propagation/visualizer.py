import os
import networkx as nx

class PropagationVisualizer:
    @staticmethod
    def export_graph_to_mermaid(G: nx.DiGraph, output_path: str = "reports/propagation_graph.mmd") -> str:
        """
        Exports the NetworkX propagation DAG to a Mermaid diagram string and saves it to reports.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mermaid_lines = ["graph TD"]
        
        # Add nodes with custom labels and shapes
        for node, attrs in G.nodes(data=True):
            n_type = attrs.get("type", "unknown")
            label = attrs.get("name", node)
            if n_type == "train":
                mermaid_lines.append(f'    {node}["🚆 Train {label} (Delay: {attrs.get("current_delay", 0):.0f}m)"]')
            elif n_type == "station":
                mermaid_lines.append(f'    {node}["🚉 Station {label} (Congestion: {attrs.get("current_congestion", 0):.0f}%)"]')
            elif n_type == "track":
                mermaid_lines.append(f'    {node}["🛤️ Track {label} (Occupancy: {attrs.get("current_occupancy", 0):.0f}%)"]')

        # Add edges
        for u, v, data in G.edges(data=True):
            reason = data.get("reason", "Dependency")
            mermaid_lines.append(f'    {u} -->|"{reason}"| {v}')
            
        mermaid_content = "\n".join(mermaid_lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(mermaid_content)
            
        return mermaid_content
