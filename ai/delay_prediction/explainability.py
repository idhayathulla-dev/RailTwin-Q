import os
import shap
import matplotlib
matplotlib.use('Agg') # Non-interactive backend to prevent GUI errors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ai.delay_prediction.utils import train_logger

class ExplainabilityEngine:
    @staticmethod
    def generate_shap_plots(model, X_train, X_test, target_name: str, output_dir: str = "reports/explainability") -> str:
        """
        Computes SHAP values on a sample of X_test and saves visual plots (Summary, Feature Importance, Waterfall).
        """
        train_logger.info(f"Generating SHAP explainability plots for {target_name}...")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Downsample test data to keep SHAP calculations very fast
            sample_size = min(200, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            
            # 1. Initialize TreeExplainer with fallback to model-agnostic Explainer
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_sample)
            except Exception as tree_err:
                train_logger.warning(f"TreeExplainer failed: {tree_err}. Falling back to black-box Explainer...")
                # We restrict the sample size to 30 to make model-agnostic kernel perturbation extremely fast
                small_sample_size = min(30, len(X_test))
                X_sample = X_test.sample(n=small_sample_size, random_state=42)
                explainer = shap.Explainer(model.predict, X_sample)
                shap_values = explainer(X_sample)
            
            # Note: For LightGBM/XGBoost, shap_values is a Explanation object.
            # We check if it is a multi-dimensional array (e.g. for classification),
            # but since these are regressor models, it is a single output target.
            
            # 2. Summary Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, show=False)
            summary_path = os.path.join(output_dir, f"{target_name}_summary_plot.png")
            plt.title(f"SHAP Summary Plot: {target_name}", fontsize=14, pad=15)
            plt.tight_layout()
            plt.savefig(summary_path, dpi=150)
            plt.close()
            train_logger.info(f"Saved SHAP summary plot to {summary_path}")

            # 3. Feature Importance Bar Plot
            plt.figure(figsize=(10, 6))
            shap.plots.bar(shap_values, max_display=20, show=False)
            bar_path = os.path.join(output_dir, f"{target_name}_feature_importance_plot.png")
            plt.title(f"SHAP Feature Importance (Top 20): {target_name}", fontsize=14, pad=15)
            plt.tight_layout()
            plt.savefig(bar_path, dpi=150)
            plt.close()
            train_logger.info(f"Saved SHAP importance plot to {bar_path}")

            # 4. Waterfall Plot (for first sample in X_sample)
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], max_display=15, show=False)
            waterfall_path = os.path.join(output_dir, f"{target_name}_waterfall_plot.png")
            plt.title(f"SHAP Waterfall (Sample 0): {target_name}", fontsize=14, pad=15)
            plt.tight_layout()
            plt.savefig(waterfall_path, dpi=150)
            plt.close()
            train_logger.info(f"Saved SHAP waterfall plot to {waterfall_path}")

            # 5. Force Plot (HTML export)
            # Standard force plots require javascript. We can export it as a standalone HTML file
            # using shap.save_html which works brilliantly.
            try:
                # We use expected_value of the explainer and shap values
                force_path = os.path.join(output_dir, f"{target_name}_force_plot.html")
                # Use standard force plot javascript rendering
                # For single sample:
                shap_val_array = shap_values.values[0]
                base_value = shap_values.base_values[0]
                # If base_value is array-like (length 1), take float
                if isinstance(base_value, np.ndarray):
                    base_value = base_value[0]
                
                shap.save_html(
                    force_path, 
                    shap.force_plot(
                        base_value, 
                        shap_val_array, 
                        X_sample.iloc[0], 
                        matplotlib=False, 
                        show=False
                    )
                )
                train_logger.info(f"Saved SHAP force plot HTML to {force_path}")
            except Exception as e:
                train_logger.warning(f"Could not save force plot HTML: {e}")

            # Compile feature importance table to export as CSV
            try:
                vals = np.abs(shap_values.values).mean(0)
                feature_importance = pd.DataFrame(
                    list(zip(X_sample.columns, vals)),
                    columns=['col_name', 'feature_importance_vals']
                )
                feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
                
                # Export csv file
                csv_path = os.path.join(output_dir, f"{target_name}_feature_importance.csv")
                feature_importance.to_csv(csv_path, index=False)
                train_logger.info(f"Saved SHAP feature importance CSV to {csv_path}")
            except Exception as e:
                train_logger.warning(f"Could not write feature importance CSV: {e}")

        except Exception as e:
            train_logger.error(f"SHAP generation error for {target_name}: {e}")
            import traceback
            train_logger.error(traceback.format_exc())
            
        return output_dir
