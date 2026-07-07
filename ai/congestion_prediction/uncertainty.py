import numpy as np

class UncertaintyEstimator:
    @staticmethod
    def calculate_confidence_and_interval(predictions_list: list, target_mean: float, scaling_factor: float = 8.0) -> tuple:
        """
        Calculates prediction mean, ensemble standard deviation, confidence score,
        and 95% prediction intervals from a list of model predictions.
        """
        predictions_array = np.array(predictions_list)
        
        # Predicted mean (aggregated final output)
        pred_mean = float(np.mean(predictions_array))
        
        # Standard deviation (disagreement between ensemble trees/folds)
        std_val = float(np.std(predictions_array))
        
        # Confidence score mapping: exponential decay of uncertainty standard deviation
        confidence = np.exp(-std_val / scaling_factor)
        confidence = max(0.40, min(0.98, confidence))
        
        # 95% Prediction Interval (mean +/- 1.96 * std)
        lower_bound = pred_mean - 1.96 * std_val
        upper_bound = pred_mean + 1.96 * std_val
        
        return pred_mean, std_val, confidence, [lower_bound, upper_bound]
