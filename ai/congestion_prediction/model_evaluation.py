import os
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ai.congestion_prediction.utils import eval_logger

class EvaluationEngine:
    @staticmethod
    def calculate_metrics(y_true, y_pred) -> dict:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0
        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "MAPE": round(mape, 2)
        }

    @staticmethod
    def generate_congestion_report_v2(results_by_target: dict, prev_registry: dict, output_dir: str = "reports") -> str:
        """
        Creates an updated evaluation report HTML comparing v2 features performance against
        v1 features recorded in prev_registry.
        """
        eval_logger.info("Generating Congestion Prediction Evaluation Report v2...")
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "congestion_evaluation_report.html")

        generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # helper to map target names to registry keys
        def get_registry_key(target: str) -> str:
            if "station_congestion" in target:
                horizon = target.split("_")[-1]
                return f"station_{horizon}_v1"
            elif "track_occupancy" in target:
                horizon = target.split("_")[-1]
                return f"track_{horizon}_v1"
            else:
                # E.g. target_network_congestion_15 -> network_network_congestion_15_v1
                target_root = target.replace("target_", "")
                return f"network_{target_root}_v1"

        comparison_cards = ""
        comparison_tables = ""

        # R2 Improvements Tracking Table
        comparison_rows_summary = ""

        for target, info in results_by_target.items():
            best_model_name = info["best_model"]
            best_metrics = info["models"]["Ensemble (Selected)"]
            
            # Look up previous stats in loaded registry
            reg_key = get_registry_key(target)
            prev_metrics = prev_registry.get(reg_key, {}).get("evaluation_metrics", {}).get("Ensemble (Selected)", {})
            
            prev_r2 = prev_metrics.get("R2", 0.0)
            prev_mae = prev_metrics.get("MAE", 999.0)
            
            r2_diff = best_metrics["R2"] - prev_r2
            mae_diff = prev_mae - best_metrics["MAE"] if prev_mae != 999.0 else 0.0

            r2_diff_color = "var(--accent-green)" if r2_diff > 0.01 else ("var(--accent-red)" if r2_diff < -0.01 else "var(--text-muted)")
            mae_diff_color = "var(--accent-green)" if mae_diff > 0.01 else ("var(--accent-red)" if mae_diff < -0.01 else "var(--text-muted)")

            comparison_rows_summary += f"""
            <tr>
                <td><strong>{target.replace("target_", "").upper()}</strong></td>
                <td>{prev_r2:.4f}</td>
                <td><strong>{best_metrics["R2"]:.4f}</strong></td>
                <td style='color: {r2_diff_color}; font-weight: 600;'>{r2_diff:+.4f}</td>
                <td>{f"{prev_mae:.4f}" if prev_mae != 999.0 else "N/A"}</td>
                <td><strong>{best_metrics["MAE"]:.4f}</strong></td>
                <td style='color: {mae_diff_color}; font-weight: 600;'>{mae_diff:+.4f}</td>
            </tr>
            """

            # Target card
            comparison_cards += f"""
            <div class="card">
                <h2>Target Model: <code>{target}</code></h2>
                <p class="subtitle">Selected Regressor: <strong>{best_model_name} (3-Fold Ensemble)</strong></p>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-val">{best_metrics["MAE"]:.4f}</div>
                        <div class="metric-lbl">MAE</div>
                        <div class="metric-diff" style='color: {mae_diff_color};'>Prev: {f"{prev_mae:.4f}" if prev_mae != 999.0 else "N/A"} ({mae_diff:+.4f})</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{best_metrics["RMSE"]:.4f}</div>
                        <div class="metric-lbl">RMSE</div>
                        <div class="metric-diff">Prev: {prev_metrics.get("RMSE", 0.0):.4f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{best_metrics["R2"]:.4f}</div>
                        <div class="metric-lbl">R-squared (R²)</div>
                        <div class="metric-diff" style='color: {r2_diff_color};'>Prev: {prev_r2:.4f} ({r2_diff:+.4f})</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{best_metrics["MAPE"]:.2f}%</div>
                        <div class="metric-lbl">MAPE</div>
                        <div class="metric-diff">Prev: {prev_metrics.get("MAPE", 0.0):.1f}%</div>
                    </div>
                </div>
            </div>
            """

            # Baseline comparisons table
            rows = ""
            for m_name, m_metrics in info["models"].items():
                is_selected = "class='best-model-row'" if m_name == "Ensemble (Selected)" else ""
                rows += f"""
                <tr {is_selected}>
                    <td><strong>{m_name}</strong></td>
                    <td>{m_metrics["MAE"]:.4f}</td>
                    <td>{m_metrics["RMSE"]:.4f}</td>
                    <td>{m_metrics["R2"]:.4f}</td>
                    <td>{m_metrics["MAPE"]:.2f}%</td>
                    <td>{"🏆 Selected" if m_name == "Ensemble (Selected)" else "Benchmark"}</td>
                </tr>
                """

            comparison_tables += f"""
            <div class="card">
                <h2>Classifier Benchmark comparison: <code>{target}</code></h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model Choice</th>
                            <th>MAE</th>
                            <th>RMSE</th>
                            <th>R²</th>
                            <th>MAPE</th>
                            <th>Selection Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>RailTwin-Q: AI Congestion Predictor Evaluation Report v2</title>
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

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-top: 20px;
        }}

        .metric-box {{
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}

        .metric-val {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--accent-purple);
            margin-bottom: 5px;
        }}

        .metric-lbl {{
            color: var(--text-muted);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .metric-diff {{
            font-size: 0.75rem;
            margin-top: 3px;
            color: var(--text-muted);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}

        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
        }}

        .best-model-row {{
            background-color: rgba(16, 185, 129, 0.08);
            border-left: 4px solid var(--accent-green);
        }}

        .best-model-row td {{
            color: #fff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>RailTwin-Q: AI Congestion Predictor Evaluation Report v2</h1>
            <p class="subtitle">Comparing Feature Engineering v2 Upgrades vs v1 Baselines</p>
        </header>

        <div class="card">
            <h2>Subsystem Performance Comparisons (v1 vs v2)</h2>
            <p>This dashboard compares the stacked learning estimators trained on the <strong>original feature set (v1)</strong> versus the <strong>expanded physical lookahead feature set (v2)</strong> (incorporating ETA windows, downstream congestion pressures, spatial centralities, and net flows).</p>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 15px;">Report Generated: {generated_at}</p>
        </div>

        <div class="card">
            <h2>R² & MAE Performance Progress Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Target Model</th>
                        <th>Prev R² (v1)</th>
                        <th>New R² (v2)</th>
                        <th>R² Change</th>
                        <th>Prev MAE</th>
                        <th>New MAE</th>
                        <th>MAE Change</th>
                    </tr>
                </thead>
                <tbody>
                    {comparison_rows_summary}
                </tbody>
            </table>
        </div>

        {comparison_cards}
        {comparison_tables}
    </div>
</body>
</html>
"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        eval_logger.info(f"Report written successfully to {report_path}")
        return report_path
