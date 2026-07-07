from ai.delay_propagation.predictor import DelayPropagationPredictor

class DisruptionPropagationEngine:
    def __init__(self, data_dir="datasets"):
        self.predictor = DelayPropagationPredictor(data_dir=data_dir)

    def evaluate_propagation(self, network, tick: int, time_str: str, active_events: list, delay_preds: list, congestion_preds: dict) -> dict:
        """
        Wrapper interface to run propagation prediction cascade analysis.
        """
        return self.predictor.get_predictions_for_tick(
            network, tick, time_str, active_events, delay_preds, congestion_preds
        )
