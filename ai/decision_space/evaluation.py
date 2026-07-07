import os
import datetime
import json

class DecisionEvaluationEngine:
    @staticmethod
    def generate_decision_report(data_dir: str = "datasets", reports_dir: str = "reports") -> str:
        """
        Compiles the Decision Intelligence evaluation report.
        """
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, "decision_evaluation_report.html")

        # Load values from datasets logs
        impact_path = os.path.join(data_dir, "decision_impact_graph.json")
        bundle_path = os.path.join(data_dir, "scenario_bundles.json")
        score_path = os.path.join(data_dir, "decision_scores.json")

        impacts = {}
        bundles = []
        scores = []

        if os.path.exists(impact_path):
            try:
                with open(impact_path, "r", encoding="utf-8") as f:
                    impacts = json.load(f)
            except Exception:
                pass
        if os.path.exists(bundle_path):
            try:
                with open(bundle_path, "r", encoding="utf-8") as f:
                    bundles = json.load(f)
            except Exception:
                pass
        if os.path.exists(score_path):
            try:
                with open(score_path, "r", encoding="utf-8") as f:
                    scores = json.load(f)
            except Exception:
                pass

        generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build actions rows
        actions_rows = ""
        for idx, (aid, detail) in enumerate(impacts.items()):
            expected = detail["expected_effects"]
            actions_rows += f"""
            <tr style="border-bottom: 1px dashed rgba(255,255,255,0.05);">
                <td><strong>{detail['action_id']}</strong></td>
                <td><span class="badge badge-blue">{detail['action']}</span></td>
                <td>{detail['target']}</td>
                <td style="color: var(--accent-green); font-weight:600;">-{expected['delay_reduction_minutes']:.1f}m</td>
                <td>{expected['expected_recovery_reduction']}m</td>
                <td>{detail['explanation']}</td>
            </tr>
            """
        if not actions_rows:
            actions_rows = "<tr><td colspan='6' style='color: var(--text-muted); text-align:center;'>No optimization candidates logged.</td></tr>"

        # Build bundle rows
        bundle_rows = ""
        for b in bundles:
            bundle_rows += f"""
            <tr style="border-bottom: 1px dashed rgba(255,255,255,0.05);">
                <td><strong>{b['bundle_id']}</strong></td>
                <td>{b['name']}</td>
                <td>{", ".join([str(x) for x in b['actions_included']]) if b['actions_included'] else 'None'}</td>
                <td><strong>{b['expected_recovery_time_mins']} mins</strong></td>
                <td style="color: var(--accent-green); font-weight:600;">-{b['total_delay_savings_mins']:.1f}m</td>
                <td><span class="badge badge-green">FEASIBLE</span></td>
            </tr>
            """
        if not bundle_rows:
            bundle_rows = "<tr><td colspan='6' style='color: var(--text-muted); text-align:center;'>No scenario bundles generated.</td></tr>"

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>RailTwin-Q: Decision Intelligence Evaluation Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-purple: #8b5cf6;
            --accent-indigo: #6366f1;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-yellow: #f59e0b;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 40px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}

        h1 {{
            margin: 0;
            font-size: 2.2rem;
            background: linear-gradient(135deg, #a855f7, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .subtitle {{
            color: var(--text-muted);
            margin: 5px 0 0 0;
            font-size: 1.1rem;
        }}

        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            backdrop-filter: blur(10px);
            margin-bottom: 25px;
        }}

        .card h2 {{
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 1.25rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
            color: var(--accent-indigo);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            text-align: left;
            margin-top: 15px;
        }}

        th {{
            color: var(--text-muted);
            border-bottom: 2px solid var(--border-color);
            padding: 10px;
        }}

        td {{
            padding: 12px 10px;
        }}

        .badge {{
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .badge-green {{ background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }}
        .badge-blue {{ background: rgba(99, 102, 241, 0.15); color: var(--accent-indigo); border: 1px solid var(--accent-indigo); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Decision Intelligence space evaluation</h1>
            <p class="subtitle">Action search space validation metrics for Layer 5 Quantum optimizer</p>
        </header>

        <div class="card">
            <h2>1. Candidate Action Search Space (Quantified Downstream Impacts)</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Action</th>
                        <th>Target</th>
                        <th>Delay reduction</th>
                        <th>Recovery reduction</th>
                        <th>Explainability & Trade-off details</th>
                    </tr>
                </thead>
                <tbody>
                    {actions_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>2. Scenario Bundles (Constraint-Satisfying Co-optimizations)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Bundle ID</th>
                        <th>Name</th>
                        <th>Actions Included</th>
                        <th>Expected Recovery Time</th>
                        <th>Global Delay Saving</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {bundle_rows}
                </tbody>
            </table>
        </div>

        <p style="color: var(--text-muted); font-size: 0.85rem; text-align: center; margin-top: 40px;">
            Report generated at {generated_at} | RailTwin-Q Layer 4 Decision Space Subsystem
        </p>
    </div>
</body>
</html>
"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return report_path
