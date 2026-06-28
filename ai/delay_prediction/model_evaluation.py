import os
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ai.delay_prediction.utils import eval_logger

class EvaluationEngine:
    @staticmethod
    def calculate_metrics(y_true, y_pred) -> dict:
        """
        Calculates regression performance metrics (MAE, RMSE, R2, MAPE).
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Custom MAPE to avoid division by zero or inflated scores for target = 0
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0
        
        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "MAPE": round(mape, 2)
        }

    @staticmethod
    def generate_report(results_by_target: dict, output_dir: str = "reports") -> str:
        """
        Creates a premium dark-themed HTML report summarizing performance metrics
        across targets and benchmarks.
        """
        eval_logger.info("Generating model evaluation report...")
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "evaluation_report.html")

        generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Compile HTML sections
        target_cards_html = ""
        comparison_tables_html = ""

        for target, info in results_by_target.items():
            best_model_name = info["best_model"]
            best_metrics = info["models"][best_model_name]
            
            # Format Card for Target
            target_cards_html += f"""
            <div class="card">
                <h2>Target Model: <code>{target}</code></h2>
                <p class="subtitle">Best Classifier: <strong>{best_model_name} Regressor</strong></p>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-val">{best_metrics["MAE"]:.4f}</div>
                        <div class="metric-lbl">Mean Absolute Error (MAE)</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{best_metrics["RMSE"]:.4f}</div>
                        <div class="metric-lbl">Root Mean Squared Error (RMSE)</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{best_metrics["R2"]:.4f}</div>
                        <div class="metric-lbl">R-squared (R²)</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-val">{best_metrics["MAPE"]:.2f}%</div>
                        <div class="metric-lbl">Mean Absolute Percentage Error (MAPE)</div>
                    </div>
                </div>
            </div>
            """

            # Format Comparison Table
            comparison_rows = ""
            for m_name, m_metrics in info["models"].items():
                is_best = "class='best-model-row'" if m_name == best_model_name else ""
                comparison_rows += f"""
                <tr {is_best}>
                    <td><strong>{m_name} Regressor</strong></td>
                    <td>{m_metrics["MAE"]:.4f}</td>
                    <td>{m_metrics["RMSE"]:.4f}</td>
                    <td>{m_metrics["R2"]:.4f}</td>
                    <td>{m_metrics["MAPE"]:.2f}%</td>
                    <td>{"🏆 Selected" if m_name == best_model_name else "Benchmark"}</td>
                </tr>
                """

            comparison_tables_html += f"""
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
                        {comparison_rows}
                    </tbody>
                </table>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>RailTwin-Q: AI Delay Predictor Evaluation Report</title>
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
            <h1>RailTwin-Q: AI Prediction Engine Evaluation</h1>
            <p class="subtitle">Detailed performance evaluation curves, benchmarks, and target validation</p>
        </header>

        <div class="card">
            <h2>Overview</h2>
            <p>This report documents the performance evaluation of the three separate regression predictors for delay forecasting at intervals of 15 minutes, 30 minutes, and 60 minutes. Models were trained on the synthetic dataset, comparing <strong>LightGBM</strong> and <strong>XGBoost</strong> classifiers, and automatically selecting the best performer.</p>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 15px;">Generated At: {generated_at}</p>
        </div>

        <!-- Cards -->
        {target_cards_html}

        <!-- Benchmark Tables -->
        {comparison_tables_html}
    </div>
</body>
</html>
"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        eval_logger.info(f"Report written successfully to {report_path}")
        return report_path
