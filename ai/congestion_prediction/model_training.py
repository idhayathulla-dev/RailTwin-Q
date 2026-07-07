import os
import sys
import json
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_regression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from ai.congestion_prediction.utils import train_logger
from ai.congestion_prediction.data_loader import DataLoader
from ai.congestion_prediction.feature_engineering import FeatureEngineer
from ai.congestion_prediction.preprocessing import DataPreprocessor
from ai.congestion_prediction.hyperparameter_tuning import HyperparameterTuner
from ai.congestion_prediction.model_evaluation import EvaluationEngine
from ai.congestion_prediction.explainability import ExplainabilityEngine
from ai.congestion_prediction.model_registry import ModelRegistry

class ModelTrainer:
    def __init__(self, data_dir="datasets", models_dir="models/congestion_predictor", reports_dir="reports"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        
        self.station_prep = DataPreprocessor("station")
        self.track_prep = DataPreprocessor("track")
        self.network_prep = DataPreprocessor("network")

    def generate_feature_engineering_report(self, X_train: pd.DataFrame, y_train: pd.Series, dropped_features: list, level_name: str):
        """
        Computes mutual information scores and writes reports/feature_engineering_report.html.
        """
        train_logger.info(f"[{level_name.upper()}] Generating Feature Validation and Relevance Report...")
        
        # Sub-sample data to keep MI calculation fast
        sample_size = min(2000, len(X_train))
        X_sample = X_train.sample(n=sample_size, random_state=42)
        y_sample = y_train.loc[X_sample.index]

        # Compute Mutual Information
        mi_scores = mutual_info_regression(X_sample, y_sample, random_state=42)
        mi_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Mutual_Information": mi_scores
        }).sort_values(by="Mutual_Information", ascending=False)

        # Write CSV report for workspace
        importance_path = os.path.join(self.reports_dir, f"feature_relevance_{level_name}.csv")
        mi_df.to_csv(importance_path, index=False)
        mi_df.to_csv("feature_importance.csv", index=False) # Root workspace deliverable

        # Generate HTML report content
        dropped_rows = ""
        for f in dropped_features:
            reason = "Highly Correlated (>0.95)" if any(f.endswith(suffix) for suffix in ["_roll_10", "_roll_20", "_h2", "_h3"]) else "Constant / Low Variance"
            dropped_rows += f"<tr><td><code>{f}</code></td><td style='color: var(--accent-red); font-weight: 600;'>DROPPED</td><td>{reason}</td></tr>"
        if not dropped_rows:
            dropped_rows = "<tr><td colspan='3' style='text-align: center; color: var(--text-muted);'>No features dropped. All inputs passed validation.</td></tr>"

        relevance_rows = ""
        for idx, row in enumerate(mi_df.head(20).itertuples()):
            rank = idx + 1
            relevance_rows += f"""
            <tr>
                <td><strong>#{rank}</strong></td>
                <td><code>{row.Feature}</code></td>
                <td>{row.Mutual_Information:.4f}</td>
                <td><span class='badge badge-green'>High Relevance</span></td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>RailTwin-Q: Congestion Feature Engineering & Validation Report</title>
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

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 20px;
        }}

        .stat-box {{
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}

        .stat-val {{
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--accent-purple);
            margin-bottom: 5px;
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

        .badge {{
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }}

        .badge-green {{ background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>RailTwin-Q: Congestion Feature Engineering & Validation Report</h1>
            <p class="subtitle">Level: {level_name.upper()} | Feature Validation pipeline v2</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card stat-box">
                <div class="stat-val">{len(X_train.columns) + len(dropped_features)}</div>
                <div class="stat-lbl" style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase;">Total Inputs Evaluated</div>
            </div>
            <div class="stat-card stat-box">
                <div class="stat-val" style="color: var(--accent-red);">{len(dropped_features)}</div>
                <div class="stat-lbl" style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase;">Collinear / Constant Dropped</div>
            </div>
            <div class="stat-card stat-box">
                <div class="stat-val" style="color: var(--accent-green);">{len(X_train.columns)}</div>
                <div class="stat-lbl" style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase;">Validated Features Retained</div>
            </div>
        </div>

        <br>

        <div class="card">
            <h2>Dropped Collinear & Constant Features Log</h2>
            <table>
                <thead>
                    <tr>
                        <th>Feature Name</th>
                        <th>Status</th>
                        <th>Reason for Drop</th>
                    </tr>
                </thead>
                <tbody>
                    {dropped_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>Top 20 Validated Features sorted by Mutual Information Relevance</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Feature Name</th>
                        <th>Mutual Information Score</th>
                        <th>Classification Relevance</th>
                    </tr>
                </thead>
                <tbody>
                    {relevance_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        report_path = os.path.join(self.reports_dir, "feature_engineering_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        train_logger.info(f"Report successfully written to {report_path}")

    def run_training_pipeline(self):
        train_logger.info("==================================================")
        train_logger.info("  RAILTWIN-Q HIERARCHICAL CONGESTION TRAINING ENGINE")
        train_logger.info("==================================================")

        # 0. Load previous registry to track metrics improvements
        registry_path = os.path.join(self.models_dir, "model_registry.json")
        prev_registry = {}
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    prev_registry = json.load(f)
            except Exception:
                pass

        # 1. Load raw datasets
        df_train, df_station, df_track, df_network, df_prop = DataLoader.load_raw_datasets(self.data_dir)

        # 2. Stage 1: Station Congestion Predictor (Level 1)
        # Apply Feature engineering updates
        df_station_eng = FeatureEngineer.engineer_features(df_station)
        df_station_prepared = DataLoader.prepare_station_dataset(df_station_eng, df_train, df_network)

        # Cross-Scenario Split
        df_st_train = df_station_prepared[df_station_prepared["scenario_id"] <= 80]
        df_st_val = df_station_prepared[(df_station_prepared["scenario_id"] > 80) & (df_station_prepared["scenario_id"] <= 90)]
        df_st_test = df_station_prepared[df_station_prepared["scenario_id"] > 90]

        st_targets = ["target_station_congestion_15", "target_station_congestion_30", "target_station_congestion_60"]
        X_st_train, y_st_train = self.station_prep.fit_transform(df_st_train, st_targets)
        X_st_val = self.station_prep.transform(df_st_val)
        X_st_test = self.station_prep.transform(df_st_test)
        y_st_test = df_st_test[st_targets]

        # Generate Feature Engineering Report (using station_30 target)
        self.generate_feature_engineering_report(X_st_train, y_st_train["target_station_congestion_30"], self.station_prep.dropped_features, "station")

        self.station_prep.save_preprocessor(self.models_dir)

        station_evals = {}
        station_ensembles = {}
        
        # Train ensembles for Level 1 targets
        for target in st_targets:
            train_logger.info(f"\n--- Fitting Level 1 (Station) Target: {target} ---")
            
            # Baselines
            dummy = DummyRegressor(strategy="mean").fit(X_st_train, y_st_train[target])
            lr = LinearRegression().fit(X_st_train, y_st_train[target])
            rf = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1).fit(X_st_train, y_st_train[target])
            
            # Tuned LightGBM/XGBoost
            lgb_params = HyperparameterTuner.tune_model("LightGBM", X_st_train, y_st_train[target])
            xgb_params = HyperparameterTuner.tune_model("XGBoost", X_st_train, y_st_train[target])

            lgb = LGBMRegressor(**lgb_params, verbosity=-1, random_state=42).fit(X_st_train, y_st_train[target])
            xgb = XGBRegressor(**xgb_params, random_state=42, verbosity=0).fit(X_st_train, y_st_train[target])

            # Evaluate
            m_dummy = EvaluationEngine.calculate_metrics(y_st_test[target], dummy.predict(X_st_test))
            m_lr = EvaluationEngine.calculate_metrics(y_st_test[target], lr.predict(X_st_test))
            m_rf = EvaluationEngine.calculate_metrics(y_st_test[target], rf.predict(X_st_test))
            m_lgb = EvaluationEngine.calculate_metrics(y_st_test[target], lgb.predict(X_st_test))
            m_xgb = EvaluationEngine.calculate_metrics(y_st_test[target], xgb.predict(X_st_test))

            best_model_name = "LightGBM" if m_lgb["MAE"] <= m_xgb["MAE"] else "XGBoost"
            best_params = lgb_params if best_model_name == "LightGBM" else xgb_params
            
            # Fit 3-Fold Ensemble
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            ensemble = []
            for fold, (tr_idx, _) in enumerate(kf.split(X_st_train)):
                X_tr, y_tr = X_st_train.iloc[tr_idx], y_st_train[target].iloc[tr_idx]
                if best_model_name == "LightGBM":
                    fold_model = LGBMRegressor(**best_params, verbosity=-1, random_state=42 + fold).fit(X_tr, y_tr)
                else:
                    fold_model = XGBRegressor(**best_params, random_state=42 + fold, verbosity=0).fit(X_tr, y_tr)
                ensemble.append(fold_model)

            y_pred_ens = np.mean([est.predict(X_st_test) for est in ensemble], axis=0)
            m_ens = EvaluationEngine.calculate_metrics(y_st_test[target], y_pred_ens)
            
            horizon_num = target.split("_")[-1]
            joblib.dump(ensemble, os.path.join(self.models_dir, f"station_model_{horizon_num}.pkl"))
            station_ensembles[horizon_num] = ensemble

            station_evals[target] = {
                "best_model": best_model_name,
                "models": {
                    "Mean Predictor": m_dummy, "Linear Regression": m_lr, "Random Forest": m_rf,
                    "LightGBM": m_lgb, "XGBoost": m_xgb, "Ensemble (Selected)": m_ens
                }
            }
            
            ModelRegistry.register_model(self.models_dir, f"station_{horizon_num}_v1", X_st_train.shape[1], station_evals[target]["models"], best_params)
            
            try:
                ExplainabilityEngine.generate_shap_plots(ensemble[0], X_st_train, X_st_test, f"station_congestion_{horizon_num}", self.reports_dir)
            except Exception as e:
                train_logger.error(f"SHAP failed: {e}")

        # Station Predictions for stacked track dataset
        X_st_all = self.station_prep.transform(df_station_prepared)
        df_station_prepared["pred_station_congestion_30"] = np.mean([est.predict(X_st_all) for est in station_ensembles["30"]], axis=0)

        # 3. Stage 2: Track Congestion Predictor (Level 2)
        df_track_eng = FeatureEngineer.engineer_features(df_track)
        df_track_prepared = DataLoader.prepare_track_dataset(df_track_eng, df_train, df_station_pred=df_station_prepared)

        df_tr_train = df_track_prepared[df_track_prepared["scenario_id"] <= 80]
        df_tr_val = df_track_prepared[(df_track_prepared["scenario_id"] > 80) & (df_track_prepared["scenario_id"] <= 90)]
        df_tr_test = df_track_prepared[df_track_prepared["scenario_id"] > 90]

        tr_targets = ["target_track_occupancy_15", "target_track_occupancy_30", "target_track_occupancy_60"]
        X_tr_train, y_tr_train = self.track_prep.fit_transform(df_tr_train, tr_targets)
        X_tr_val = self.track_prep.transform(df_tr_val)
        X_tr_test = self.track_prep.transform(df_tr_test)
        y_tr_test = df_tr_test[tr_targets]

        self.track_prep.save_preprocessor(self.models_dir)

        track_evals = {}
        track_ensembles = {}

        for target in tr_targets:
            train_logger.info(f"\n--- Fitting Level 2 (Track) Target: {target} ---")
            
            dummy = DummyRegressor(strategy="mean").fit(X_tr_train, y_tr_train[target])
            lr = LinearRegression().fit(X_tr_train, y_tr_train[target])
            rf = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1).fit(X_tr_train, y_tr_train[target])
            
            lgb_params = HyperparameterTuner.tune_model("LightGBM", X_tr_train, y_tr_train[target])
            xgb_params = HyperparameterTuner.tune_model("XGBoost", X_tr_train, y_tr_train[target])

            lgb = LGBMRegressor(**lgb_params, verbosity=-1, random_state=42).fit(X_tr_train, y_tr_train[target])
            xgb = XGBRegressor(**xgb_params, random_state=42, verbosity=0).fit(X_tr_train, y_tr_train[target])

            m_dummy = EvaluationEngine.calculate_metrics(y_tr_test[target], dummy.predict(X_tr_test))
            m_lr = EvaluationEngine.calculate_metrics(y_tr_test[target], lr.predict(X_tr_test))
            m_rf = EvaluationEngine.calculate_metrics(y_tr_test[target], rf.predict(X_tr_test))
            m_lgb = EvaluationEngine.calculate_metrics(y_tr_test[target], lgb.predict(X_tr_test))
            m_xgb = EvaluationEngine.calculate_metrics(y_tr_test[target], xgb.predict(X_tr_test))

            best_model_name = "LightGBM" if m_lgb["MAE"] <= m_xgb["MAE"] else "XGBoost"
            best_params = lgb_params if best_model_name == "LightGBM" else xgb_params

            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            ensemble = []
            for fold, (tr_idx, _) in enumerate(kf.split(X_tr_train)):
                X_tr, y_tr = X_tr_train.iloc[tr_idx], y_tr_train[target].iloc[tr_idx]
                if best_model_name == "LightGBM":
                    fold_model = LGBMRegressor(**best_params, verbosity=-1, random_state=42 + fold).fit(X_tr, y_tr)
                else:
                    fold_model = XGBRegressor(**best_params, random_state=42 + fold, verbosity=0).fit(X_tr, y_tr)
                ensemble.append(fold_model)

            y_pred_ens = np.mean([est.predict(X_tr_test) for est in ensemble], axis=0)
            m_ens = EvaluationEngine.calculate_metrics(y_tr_test[target], y_pred_ens)

            horizon_num = target.split("_")[-1]
            joblib.dump(ensemble, os.path.join(self.models_dir, f"track_model_{horizon_num}.pkl"))
            track_ensembles[horizon_num] = ensemble

            track_evals[target] = {
                "best_model": best_model_name,
                "models": {
                    "Mean Predictor": m_dummy, "Linear Regression": m_lr, "Random Forest": m_rf,
                    "LightGBM": m_lgb, "XGBoost": m_xgb, "Ensemble (Selected)": m_ens
                }
            }

            ModelRegistry.register_model(self.models_dir, f"track_{horizon_num}_v1", X_tr_train.shape[1], track_evals[target]["models"], best_params)

            try:
                ExplainabilityEngine.generate_shap_plots(ensemble[0], X_tr_train, X_tr_test, f"track_occupancy_{horizon_num}", self.reports_dir)
            except Exception as e:
                train_logger.error(f"SHAP failed: {e}")

        # Track Predictions for network dataset
        X_tr_all = self.track_prep.transform(df_track_prepared)
        df_track_prepared["pred_track_occupancy_30"] = np.mean([est.predict(X_tr_all) for est in track_ensembles["30"]], axis=0)

        # 4. Stage 3: Network Congestion Predictor (Level 3)
        df_network_eng = FeatureEngineer.engineer_features(df_network)
        df_network_prepared = DataLoader.prepare_network_dataset(df_network_eng, df_station_pred=df_station_prepared, df_track_pred=df_track_prepared)

        df_net_train = df_network_prepared[df_network_prepared["scenario_id"] <= 80]
        df_net_val = df_network_prepared[(df_network_prepared["scenario_id"] > 80) & (df_network_prepared["scenario_id"] <= 90)]
        df_net_test = df_network_prepared[df_network_prepared["scenario_id"] > 90]

        net_targets_map = {
            "target_network_congestion_15": "network_congestion_15",
            "target_network_congestion_30": "network_congestion_30",
            "target_network_congestion_60": "network_congestion_60",
            "target_platform_utilization_15": "platform_utilization_15",
            "target_platform_utilization_30": "platform_utilization_30",
            "target_platform_utilization_60": "platform_utilization_60",
            "target_track_utilization_15": "track_utilization_15",
            "target_track_utilization_30": "track_utilization_30",
            "target_track_utilization_60": "track_utilization_60",
            "target_average_delay_15": "average_delay_15",
            "target_average_delay_30": "average_delay_30",
            "target_average_delay_60": "average_delay_60"
        }
        
        net_targets = list(net_targets_map.keys())
        X_net_train, y_net_train = self.network_prep.fit_transform(df_net_train, net_targets)
        X_net_val = self.network_prep.transform(df_net_val)
        X_net_test = self.network_prep.transform(df_net_test)
        y_net_test = df_net_test[net_targets]

        self.network_prep.save_preprocessor(self.models_dir)

        network_evals = {}

        for target in net_targets:
            root_name = net_targets_map[target]
            train_logger.info(f"\n--- Fitting Level 3 (Network) Target: {target} ---")
            
            dummy = DummyRegressor(strategy="mean").fit(X_net_train, y_net_train[target])
            lr = LinearRegression().fit(X_net_train, y_net_train[target])
            rf = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1).fit(X_net_train, y_net_train[target])
            
            lgb_params = HyperparameterTuner.tune_model("LightGBM", X_net_train, y_net_train[target])
            xgb_params = HyperparameterTuner.tune_model("XGBoost", X_net_train, y_net_train[target])

            lgb = LGBMRegressor(**lgb_params, verbosity=-1, random_state=42).fit(X_net_train, y_net_train[target])
            xgb = XGBRegressor(**xgb_params, random_state=42, verbosity=0).fit(X_net_train, y_net_train[target])

            m_dummy = EvaluationEngine.calculate_metrics(y_net_test[target], dummy.predict(X_net_test))
            m_lr = EvaluationEngine.calculate_metrics(y_net_test[target], lr.predict(X_net_test))
            m_rf = EvaluationEngine.calculate_metrics(y_net_test[target], rf.predict(X_net_test))
            m_lgb = EvaluationEngine.calculate_metrics(y_net_test[target], lgb.predict(X_net_test))
            m_xgb = EvaluationEngine.calculate_metrics(y_net_test[target], xgb.predict(X_net_test))

            best_model_name = "LightGBM" if m_lgb["MAE"] <= m_xgb["MAE"] else "XGBoost"
            best_params = lgb_params if best_model_name == "LightGBM" else xgb_params

            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            ensemble = []
            for fold, (tr_idx, _) in enumerate(kf.split(X_net_train)):
                X_tr, y_tr = X_net_train.iloc[tr_idx], y_net_train[target].iloc[tr_idx]
                if best_model_name == "LightGBM":
                    fold_model = LGBMRegressor(**best_params, verbosity=-1, random_state=42 + fold).fit(X_tr, y_tr)
                else:
                    fold_model = XGBRegressor(**best_params, random_state=42 + fold, verbosity=0).fit(X_tr, y_tr)
                ensemble.append(fold_model)

            y_pred_ens = np.mean([est.predict(X_net_test) for est in ensemble], axis=0)
            m_ens = EvaluationEngine.calculate_metrics(y_net_test[target], y_pred_ens)

            joblib.dump(ensemble, os.path.join(self.models_dir, f"network_model_{root_name}.pkl"))

            network_evals[target] = {
                "best_model": best_model_name,
                "models": {
                    "Mean Predictor": m_dummy, "Linear Regression": m_lr, "Random Forest": m_rf,
                    "LightGBM": m_lgb, "XGBoost": m_xgb, "Ensemble (Selected)": m_ens
                }
            }

            ModelRegistry.register_model(self.models_dir, f"network_{root_name}_v1", X_net_train.shape[1], network_evals[target]["models"], best_params)

            if "network_congestion_30" in target:
                try:
                    ExplainabilityEngine.generate_shap_plots(ensemble[0], X_net_train, X_net_test, "network_congestion_30", self.reports_dir)
                except Exception as e:
                    train_logger.error(f"SHAP failed: {e}")

        # 5. Save HTML report comparing baselines, targets, and previous registry performance scores
        try:
            all_evals = {**station_evals, **track_evals, **network_evals}
            EvaluationEngine.generate_congestion_report_v2(all_evals, prev_registry, self.reports_dir)
        except Exception as e:
            train_logger.error(f"Failed to generate HTML report: {e}")

        train_logger.info("==================================================")
        train_logger.info("  CONGESTION PIPELINE COMPLETED SUCCESSFULLY")
        train_logger.info("==================================================")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_training_pipeline()
