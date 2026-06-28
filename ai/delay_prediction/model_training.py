import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from ai.delay_prediction.utils import train_logger
from ai.delay_prediction.data_loader import DataLoader
from ai.delay_prediction.feature_engineering import FeatureEngineer
from ai.delay_prediction.preprocessing import DataPreprocessor
from ai.delay_prediction.hyperparameter_tuning import HyperparameterTuner
from ai.delay_prediction.model_evaluation import EvaluationEngine
from ai.delay_prediction.explainability import ExplainabilityEngine
from ai.delay_prediction.model_registry import ModelRegistry

class ModelTrainer:
    def __init__(self, data_dir="datasets", models_dir="models/delay_predictor", reports_dir="reports"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.preprocessor = DataPreprocessor()

    def run_training_pipeline(self):
        train_logger.info("==================================================")
        train_logger.info("    RAILTWIN-Q AI DELAY PREDICTION TRAINING ENGINE")
        train_logger.info("==================================================")

        # 1. Load and merge data
        df = DataLoader.load_and_merge_data(self.data_dir)

        # 2. Feature Engineering
        df_engineered = FeatureEngineer.engineer_features(df)

        # 3. Cross-Scenario Splits (Train <=80, Val 81-90, Test 91-100)
        # This prevents identical simulation ticks from appearing in both training and testing
        df_train = df_engineered[df_engineered["scenario_id"] <= 80]
        df_val = df_engineered[(df_engineered["scenario_id"] > 80) & (df_engineered["scenario_id"] <= 90)]
        df_test = df_engineered[df_engineered["scenario_id"] > 90]
        
        train_logger.info(f"Cross-Scenario Split sizes: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

        # 4. Preprocess variables
        targets = ["future_delay_15", "future_delay_30", "future_delay_60"]
        X_train, y_train = self.preprocessor.fit_transform(df_train, targets)
        X_val = self.preprocessor.transform(df_val)
        y_val = df_val[targets]
        X_test = self.preprocessor.transform(df_test)
        y_test = df_test[targets]

        # Save preprocessors
        self.preprocessor.save_preprocessors(self.models_dir)

        # Store best estimators and evaluation results for report
        best_models = {}
        evaluation_results = {}
        feature_importance_df = pd.DataFrame(index=X_train.columns)

        # 5. Model building loop per target
        for target in targets:
            train_logger.info(f"\n--- Training predictors for target: {target} ---")
            
            # Step A: Fit Baseline Models
            # 1. Mean Predictor
            dummy = DummyRegressor(strategy="mean")
            dummy.fit(X_train, y_train[target])
            y_pred_mean = dummy.predict(X_test)
            mean_metrics = EvaluationEngine.calculate_metrics(y_test[target], y_pred_mean)
            
            # 2. Linear Regression
            lr = LinearRegression()
            lr.fit(X_train, y_train[target])
            y_pred_lr = lr.predict(X_test)
            lr_metrics = EvaluationEngine.calculate_metrics(y_test[target], y_pred_lr)
            
            # 3. Random Forest (restricted size to stay lightweight)
            rf = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train[target])
            y_pred_rf = rf.predict(X_test)
            rf_metrics = EvaluationEngine.calculate_metrics(y_test[target], y_pred_rf)

            # Step B: Hyperparameter tuning for Advanced Models
            lgb_params = HyperparameterTuner.tune_lightgbm(X_train, y_train[target])
            xgb_params = HyperparameterTuner.tune_xgboost(X_train, y_train[target])

            # Step C: Model instances & fit
            lgb_model = LGBMRegressor(**lgb_params, verbosity=-1, random_state=42)
            xgb_model = XGBRegressor(**xgb_params, random_state=42, verbosity=0)

            train_logger.info("Fitting LightGBM on training set...")
            lgb_model.fit(X_train, y_train[target])
            train_logger.info("Fitting XGBoost on training set...")
            xgb_model.fit(X_train, y_train[target])

            # Step D: Evaluate on Test Set
            y_pred_lgb = lgb_model.predict(X_test)
            y_pred_xgb = xgb_model.predict(X_test)

            lgb_metrics = EvaluationEngine.calculate_metrics(y_test[target], y_pred_lgb)
            xgb_metrics = EvaluationEngine.calculate_metrics(y_test[target], y_pred_xgb)

            train_logger.info(f"Baseline Mean evaluation: {mean_metrics}")
            train_logger.info(f"Linear Regression evaluation: {lr_metrics}")
            train_logger.info(f"Random Forest evaluation: {rf_metrics}")
            train_logger.info(f"LightGBM evaluation: {lgb_metrics}")
            train_logger.info(f"XGBoost evaluation: {xgb_metrics}")

            # Step E: Compare and select the best model (by MAE)
            best_model_name = "LightGBM" if lgb_metrics["MAE"] <= xgb_metrics["MAE"] else "XGBoost"
            best_params = lgb_params if best_model_name == "LightGBM" else xgb_params
            
            train_logger.info(f"🏆 Selected model type for {target}: {best_model_name}")

            # Step F: Train a 3-fold cross-validation ensemble of the best classifier
            # Used to estimate prediction variance (Bootstrap uncertainty) during inference
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            ensemble_estimators = []
            train_logger.info(f"Training 3-fold CV ensemble of {best_model_name}...")
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                X_tr, y_tr = X_train.iloc[train_idx], y_train[target].iloc[train_idx]
                if best_model_name == "LightGBM":
                    fold_model = LGBMRegressor(**best_params, verbosity=-1, random_state=42 + fold)
                else:
                    fold_model = XGBRegressor(**best_params, random_state=42 + fold, verbosity=0)
                
                fold_model.fit(X_tr, y_tr)
                ensemble_estimators.append(fold_model)

            # Evaluate ensemble mean prediction
            y_pred_ens_list = [est.predict(X_test) for est in ensemble_estimators]
            y_pred_ens = np.mean(y_pred_ens_list, axis=0)
            ensemble_metrics = EvaluationEngine.calculate_metrics(y_test[target], y_pred_ens)
            train_logger.info(f"Ensemble performance on test set: {ensemble_metrics}")

            # Save the ensemble estimators list
            model_filename = f"delay_{target.replace('future_delay_', '')}min.pkl"
            model_path = os.path.join(self.models_dir, model_filename)
            joblib.dump(ensemble_estimators, model_path)
            train_logger.info(f"Saved {best_model_name} ensemble (3 estimators) for {target} to {model_path}")

            best_models[target] = ensemble_estimators[0] # use first estimator for SHAP
            evaluation_results[target] = {
                "best_model": best_model_name,
                "models": {
                    "Mean Predictor": mean_metrics,
                    "Linear Regression": lr_metrics,
                    "Random Forest": rf_metrics,
                    "LightGBM": lgb_metrics,
                    "XGBoost": xgb_metrics,
                    "Ensemble (Selected)": ensemble_metrics
                }
            }

            # Register model version
            ModelRegistry.register_model(
                models_dir=self.models_dir,
                version=f"{target}_v1",
                feature_count=X_train.shape[1],
                metrics=evaluation_results[target]["models"],
                hyperparameters=best_params
            )

            # Step G: SHAP explainability (using first estimator in ensemble)
            try:
                shap_dir = os.path.join(self.reports_dir, "explainability")
                ExplainabilityEngine.generate_shap_plots(ensemble_estimators[0], X_train, X_test, target, shap_dir)
            except Exception as e:
                train_logger.error(f"Error generating SHAP explainability for {target}: {e}")

            # Compile feature importance values
            try:
                if best_model_name == "LightGBM":
                    importances = ensemble_estimators[0].feature_importances_
                else:
                    importances = ensemble_estimators[0].feature_importances_
                feature_importance_df[f"{target}_importance"] = importances
            except Exception as e:
                train_logger.warning(f"Could not extract feature importances: {e}")

        # 6. Save compiled feature importance CSV
        try:
            feature_importance_df["mean_importance"] = feature_importance_df.mean(axis=1)
            feature_importance_df.sort_values(by="mean_importance", ascending=False, inplace=True)
            
            csv_report_path = os.path.join(self.reports_dir, "feature_importance.csv")
            feature_importance_df.to_csv(csv_report_path)
            feature_importance_df.to_csv("feature_importance.csv")
            train_logger.info(f"Compiled feature importance csv written to {csv_report_path} and feature_importance.csv")
        except Exception as e:
            train_logger.error(f"Error compiling feature importances: {e}")

        # 7. Generate Evaluation HTML Report
        try:
            EvaluationEngine.generate_report(evaluation_results, self.reports_dir)
        except Exception as e:
            train_logger.error(f"Error generating evaluation report HTML: {e}")

        train_logger.info("==================================================")
        train_logger.info("      MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        train_logger.info("==================================================")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_training_pipeline()
