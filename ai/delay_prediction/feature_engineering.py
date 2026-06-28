import numpy as np
import pandas as pd
from ai.delay_prediction.utils import train_logger

class FeatureEngineer:
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature engineering pipelines (time cycling, complexity, weather extensions).
        """
        train_logger.info("Starting feature engineering...")
        df = df.copy()

        # 1. Cyclical Time Features
        # Extract hour and minute from 'time' column (format "HH:MM")
        try:
            df["hour"] = df["time"].apply(lambda x: int(x.split(":")[0]) if isinstance(x, str) else 8)
            df["minute"] = df["time"].apply(lambda x: int(x.split(":")[1]) if isinstance(x, str) else 0)
        except Exception as e:
            train_logger.warning(f"Error parsing time strings: {e}. Defaulting to hour 8, min 0.")
            df["hour"] = 8
            df["minute"] = 0

        # Cyclical transformations
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60.0)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60.0)

        # 2. Weather & Visibility Features
        # 'weather_type' (CLEAR, RAINY) and 'rain_intensity' (0.0 to 1.0) are already present
        df["heavy_rain"] = df["rain_intensity"].apply(lambda x: 1 if x >= 0.7 else 0)
        df["storm"] = df["rain_intensity"].apply(lambda x: 1 if x >= 0.85 else 0)
        # Visibility reduces in rain: e.g., base of 10km down to 1km in heavy rain
        df["visibility"] = df["rain_intensity"].apply(lambda x: round(10.0 * (1.0 - x * 0.9), 2))

        # 3. Route Complexity Features
        # Columns present: remaining_stations, route_length, junction_count, alternative_routes, critical_station_count
        # Let's rename or alias alternative_routes to alternative_route_count
        if "alternative_routes" in df.columns:
            df["alternative_route_count"] = df["alternative_routes"]
        else:
            df["alternative_route_count"] = 1
            
        # Composite route complexity score
        df["route_complexity_score"] = round(
            (df["route_length"] / 100.0) * 0.4 + 
            (df["junction_count"] * 0.3) + 
            (df["critical_station_count"] * 0.3), 
            2
        )

        train_logger.info("Feature engineering complete.")
        return df
