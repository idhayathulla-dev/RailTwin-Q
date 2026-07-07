import os
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ai.congestion_prediction.utils import train_logger

class ExplainabilityEngine:
    @staticmethod
    def generate_shap_plots(model, X_train, X_test, target_name: str, output_dir: str = "reports") -> str:
        """
        Computes SHAP values on a sample of X_test and saves visual plots to a congestion_explainability sub-directory.
        """
        explain_dir = os.path.join(output_dir, "congestion_explainability")
        train_logger.info(f"Generating SHAP plots for {target_name} inside {explain_dir}...")
        os.makedirs(explain_dir, exist_ok=True)
        
        try:
            # Downsample test data to keep calculations fast
            sample_size = min(150, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            
            # TreeExplainer with Explainer fallback
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_sample)
            except Exception as tree_err:
                train_logger.warning(f"TreeExplainer failed for {target_name}: {tree_err}. Falling back to Explainer...")
                small_sample_size = min(30, len(X_test))
                X_sample = X_test.sample(n=small_sample_size, random_state=42)
                explainer = shap.Explainer(model.predict, X_sample)
                shap_values = explainer(X_sample)
            
            # 1. Summary Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, show=False)
            summary_path = os.path.join(explain_dir, f"{target_name}_summary_plot.png")
            plt.title(f"SHAP Summary Plot: {target_name}", fontsize=13, pad=15)
            plt.tight_layout()
            plt.savefig(summary_path, dpi=150)
            plt.close()

            # 2. Importance Bar Plot
            plt.figure(figsize=(10, 6))
            shap.plots.bar(shap_values, max_display=15, show=False)
            bar_path = os.path.join(explain_dir, f"{target_name}_feature_importance_plot.png")
            plt.title(f"SHAP Feature Importance (Top 15): {target_name}", fontsize=13, pad=15)
            plt.tight_layout()
            plt.savefig(bar_path, dpi=150)
            plt.close()

            # 3. Waterfall Plot
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], max_display=12, show=False)
            waterfall_path = os.path.join(explain_dir, f"{target_name}_waterfall_plot.png")
            plt.title(f"SHAP Waterfall (Sample 0): {target_name}", fontsize=13, pad=15)
            plt.tight_layout()
            plt.savefig(waterfall_path, dpi=150)
            plt.close()

            # 4. Force Plot HTML
            try:
                force_path = os.path.join(explain_dir, f"{target_name}_force_plot.html")
                shap_val_array = shap_values.values[0]
                base_value = shap_values.base_values[0]
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
            except Exception as e:
                train_logger.warning(f"Could not save force plot HTML for {target_name}: {e}")

            # 5. Save importance CSV
            try:
                vals = np.abs(shap_values.values).mean(0)
                feature_importance = pd.DataFrame(
                    list(zip(X_sample.columns, vals)),
                    columns=['col_name', 'feature_importance_vals']
                )
                feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=False)
                csv_path = os.path.join(explain_dir, f"{target_name}_feature_importance.csv")
                feature_importance.to_csv(csv_path, index=False)
            except Exception as e:
                train_logger.warning(f"Could not write CSV for {target_name}: {e}")

        except Exception as e:
            train_logger.error(f"SHAP generation error for {target_name}: {e}")
            
        return explain_dir
