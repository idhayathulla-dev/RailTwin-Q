import os
import datetime
import json

class ReportGenerator:
    @staticmethod
    def generate_all_reports(
        reasoning: dict,
        counterfactuals: dict,
        cost_vectors: list,
        robustness: dict,
        pareto_front: dict,
        search_space: dict,
        reports_dir="reports"
    ):
        """
        Compiles the five specialized evaluation reports under reports/ directory.
        """
        os.makedirs(reports_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Decision Reasoning Report
        reasoning_rows = ""
        for aid, item in reasoning.items():
            reasoning_rows += f"""
            <tr style="border-bottom: 1px dashed rgba(255,255,255,0.05);">
                <td><strong>{aid}</strong></td>
                <td><span class="badge badge-blue">{item['action']}</span></td>
                <td>{item['target']}</td>
                <td><div style="font-size:0.75rem; color:var(--text-muted);">{item['explanation_text']}</div></td>
                <td><strong>{item['confidence'] * 100:.0f}%</strong></td>
            </tr>
            """
        
        reasoning_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Decision Reasoning Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{ --bg-color: #0f172a; --card-bg: rgba(30, 41, 59, 0.7); --border-color: rgba(255,255,255,0.1); --text-main: #f8fafc; --text-muted: #94a3b8; --accent-indigo: #6366f1; --accent-blue: #3b82f6; }}
            body {{ background-color: var(--bg-color); color: var(--text-main); font-family: 'Outfit', sans-serif; padding: 40px; }}
            .card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 24px; }}
            table {{ width:100%; border-collapse:collapse; text-align:left; }}
            th {{ border-bottom: 2px solid var(--border-color); padding: 10px; color: var(--text-muted); }}
            td {{ padding: 12px 10px; }}
            .badge {{ padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
            .badge-blue {{ background: rgba(99, 102, 241, 0.15); color: var(--accent-indigo); border: 1px solid var(--accent-indigo); }}
        </style></head>
        <body><div class="card">
            <h1 style="margin-top:0; color:var(--accent-indigo);">Decision Reasoning Trace Report</h1>
            <p style="color:var(--text-muted);">Causal chain evaluation pathways generated for active candidate actions.</p>
            <table>
                <thead><tr><th>ID</th><th>Action</th><th>Target</th><th>Causal Chain Reasoning</th><th>Confidence</th></tr></thead>
                <tbody>{reasoning_rows}</tbody>
            </table>
            <p style="text-align:center; font-size:0.8rem; color:var(--text-muted); margin-top:30px;">Generated at {timestamp}</p>
        </div></body></html>"""
        
        with open(os.path.join(reports_dir, "decision_reasoning_report.html"), "w", encoding="utf-8") as f:
            f.write(reasoning_html)

        # 2. Counterfactual Report
        counterfactual_rows = ""
        for name, sc in counterfactuals.items():
            counterfactual_rows += f"""
            <tr style="border-bottom: 1px dashed rgba(255,255,255,0.05);">
                <td><strong>{name}</strong></td>
                <td>{sc['recovery_time_mins']} mins</td>
                <td style="color:var(--accent-indigo); font-weight:600;">-{sc['delay_reduction_minutes']}m</td>
                <td>-{sc['congestion_reduction_percent']}%</td>
                <td>-{sc['csi_improvement']}%</td>
            </tr>
            """
        
        counterfactual_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Counterfactual Analysis Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{ --bg-color: #0f172a; --card-bg: rgba(30, 41, 59, 0.7); --border-color: rgba(255,255,255,0.1); --text-main: #f8fafc; --text-muted: #94a3b8; --accent-indigo: #6366f1; }}
            body {{ background-color: var(--bg-color); color: var(--text-main); font-family: 'Outfit', sans-serif; padding: 40px; }}
            .card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 24px; }}
            table {{ width:100%; border-collapse:collapse; text-align:left; }}
            th {{ border-bottom: 2px solid var(--border-color); padding: 10px; color: var(--text-muted); }}
            td {{ padding: 12px 10px; }}
        </style></head>
        <body><div class="card">
            <h1 style="margin-top:0; color:var(--accent-indigo);">Counterfactual Scenario Report</h1>
            <p style="color:var(--text-muted);">Comparative what-if analysis of independent operational configurations.</p>
            <table>
                <thead><tr><th>Scenario</th><th>Expected Recovery Time</th><th>Delay Reduction</th><th>Congestion Drop</th><th>CSI Improvement</th></tr></thead>
                <tbody>{counterfactual_rows}</tbody>
            </table>
            <p style="text-align:center; font-size:0.8rem; color:var(--text-muted); margin-top:30px;">Generated at {timestamp}</p>
        </div></body></html>"""

        with open(os.path.join(reports_dir, "counterfactual_report.html"), "w", encoding="utf-8") as f:
            f.write(counterfactual_html)

        # 3. Robustness Report
        robustness_rows = ""
        for aid, item in robustness.items():
            comp_html = ""
            for p, works in item["compatibility"].items():
                icon = "<span style='color:#10b981;'>✓</span>" if works else "<span style='color:#ef4444;'>✗</span>"
                comp_html += f"<span style='margin-right:12px; font-size:0.8rem;'>{p}: {icon}</span>"
            
            robustness_rows += f"""
            <tr style="border-bottom: 1px dashed rgba(255,255,255,0.05);">
                <td><strong>{aid}</strong></td>
                <td><span class="badge badge-blue">{item['action']}</span></td>
                <td>{comp_html}</td>
                <td><strong>{item['robustness_score']}%</strong></td>
            </tr>
            """

        robustness_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Robustness Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{ --bg-color: #0f172a; --card-bg: rgba(30, 41, 59, 0.7); --border-color: rgba(255,255,255,0.1); --text-main: #f8fafc; --text-muted: #94a3b8; --accent-indigo: #6366f1; }}
            body {{ background-color: var(--bg-color); color: var(--text-main); font-family: 'Outfit', sans-serif; padding: 40px; }}
            .card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 24px; }}
            table {{ width:100%; border-collapse:collapse; text-align:left; }}
            th {{ border-bottom: 2px solid var(--border-color); padding: 10px; color: var(--text-muted); }}
            td {{ padding: 12px 10px; }}
            .badge {{ padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
            .badge-blue {{ background: rgba(99, 102, 241, 0.15); color: var(--accent-indigo); border: 1px solid var(--accent-indigo); }}
        </style></head>
        <body><div class="card">
            <h1 style="margin-top:0; color:var(--accent-indigo);">Disruption Profile Robustness Report</h1>
            <p style="color:var(--text-muted);">Action applicability scores compiled under diverse simulation anomaly triggers.</p>
            <table>
                <thead><tr><th>ID</th><th>Action</th><th>Condition Applicability Matrix</th><th>Robustness Rating</th></tr></thead>
                <tbody>{robustness_rows}</tbody>
            </table>
            <p style="text-align:center; font-size:0.8rem; color:var(--text-muted); margin-top:30px;">Generated at {timestamp}</p>
        </div></body></html>"""

        with open(os.path.join(reports_dir, "robustness_report.html"), "w", encoding="utf-8") as f:
            f.write(robustness_html)

        # 4. Pareto Report
        pareto_rows = ""
        for item in pareto_front.get("pareto_solutions", []):
            pareto_rows += f"""
            <tr style="border-bottom: 1px dashed rgba(255,255,255,0.05);">
                <td><strong>{item['action_id']}</strong></td>
                <td><span class="badge badge-blue">{item['action']}</span></td>
                <td>{item['target']}</td>
                <td style="color:#ef4444;">{item['delay_cost']:.2f}</td>
                <td>{item['risk']:.2f}</td>
                <td>{item['energy']:.2f}</td>
                <td style="color:#10b981;">{item['passenger_saved_percent'] * 100:.0f}%</td>
                <td>{item['operational_cost']:.2f}</td>
                <td>{item['robustness'] * 100:.0f}%</td>
            </tr>
            """

        pareto_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Pareto Frontier Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{ --bg-color: #0f172a; --card-bg: rgba(30, 41, 59, 0.7); --border-color: rgba(255,255,255,0.1); --text-main: #f8fafc; --text-muted: #94a3b8; --accent-indigo: #6366f1; }}
            body {{ background-color: var(--bg-color); color: var(--text-main); font-family: 'Outfit', sans-serif; padding: 40px; }}
            .card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 24px; }}
            table {{ width:100%; border-collapse:collapse; text-align:left; }}
            th {{ border-bottom: 2px solid var(--border-color); padding: 10px; color: var(--text-muted); }}
            td {{ padding: 12px 10px; }}
            .badge {{ padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
            .badge-blue {{ background: rgba(99, 102, 241, 0.15); color: var(--accent-indigo); border: 1px solid var(--accent-indigo); }}
        </style></head>
        <body><div class="card">
            <h1 style="margin-top:0; color:var(--accent-indigo);">Pareto Front Co-Optimization Report</h1>
            <p style="color:var(--text-muted);">Non-dominated candidate solutions balancing delay mitigation, risk exposure, and energy footprints.</p>
            <table>
                <thead><tr><th>ID</th><th>Action</th><th>Target</th><th>Delay Cost</th><th>Risk Cost</th><th>Energy Cost</th><th>Passenger Saved</th><th>Op Cost</th><th>Robustness</th></tr></thead>
                <tbody>{pareto_rows}</tbody>
            </table>
            <p style="text-align:center; font-size:0.8rem; color:var(--text-muted); margin-top:30px;">Generated at {timestamp}</p>
        </div></body></html>"""

        with open(os.path.join(reports_dir, "pareto_report.html"), "w", encoding="utf-8") as f:
            f.write(pareto_html)

        # 5. Optimization Search Space Report
        search_rows = ""
        for idx, (var_id, detail) in enumerate(search_space["decision_variables"].items()):
            search_rows += f"""
            <tr style="border-bottom: 1px dashed rgba(255,255,255,0.05);">
                <td><strong>{detail['action_id']}</strong></td>
                <td><code>{detail['variable_symbol']}</code></td>
                <td><span class="badge badge-blue">{detail['action']}</span></td>
                <td>{detail['target']}</td>
                <td><span style="color:#10b981; font-weight:600;">FEASIBLE</span></td>
            </tr>
            """

        space_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Optimization Search Space Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{ --bg-color: #0f172a; --card-bg: rgba(30, 41, 59, 0.7); --border-color: rgba(255,255,255,0.1); --text-main: #f8fafc; --text-muted: #94a3b8; --accent-indigo: #6366f1; }}
            body {{ background-color: var(--bg-color); color: var(--text-main); font-family: 'Outfit', sans-serif; padding: 40px; }}
            .card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 24px; }}
            table {{ width:100%; border-collapse:collapse; text-align:left; }}
            th {{ border-bottom: 2px solid var(--border-color); padding: 10px; color: var(--text-muted); }}
            td {{ padding: 12px 10px; }}
            .badge {{ padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
            .badge-blue {{ background: rgba(99, 102, 241, 0.15); color: var(--accent-indigo); border: 1px solid var(--accent-indigo); }}
        </style></head>
        <body><div class="card">
            <h1 style="margin-top:0; color:var(--accent-indigo);">Optimization Search Space Formulation</h1>
            <p style="color:var(--text-muted);">Mathematical variables mapping for QUBO compiler matrices inside Layer 5.</p>
            <table>
                <thead><tr><th>Action ID</th><th>Decision Variable Symbol</th><th>Action</th><th>Target</th><th>Feasibility Status</th></tr></thead>
                <tbody>{search_rows}</tbody>
            </table>
            <p style="text-align:center; font-size:0.8rem; color:var(--text-muted); margin-top:30px;">Generated at {timestamp}</p>
        </div></body></html>"""

        with open(os.path.join(reports_dir, "optimization_search_space_report.html"), "w", encoding="utf-8") as f:
            f.write(space_html)
