import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ai.congestion_prediction.utils import train_logger

class DataPreprocessor:
    def __init__(self, level_name: str = "station"):
        self.level_name = level_name
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []
        self.boolean_cols = []
        self.feature_ordering = []
        self.dropped_features = []

    def fit_transform(self, df: pd.DataFrame, target_cols: list) -> tuple:
        """
        Fits and transforms the dataset. Automatically performs feature selection
        by removing constant, duplicate, and highly correlated features before fitting models.
        """
        train_logger.info(f"[{self.level_name.upper()}] Fitting and transforming preprocessors...")
        df = df.copy()
        
        # Safe inf/NaN replace
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Columns to exclude (resolving 30-minute target leakage from simulator generated columns)
        exclude_cols = target_cols + [
            "scenario_id", "tick", "state_id", "time", "station_name", "source_station", "destination_station",
            "active_disruptions", "weather_type", "time_of_day", "day_type",
            "future_station_congestion", "future_track_congestion", "future_network_congestion",
            "future_platform_utilization", "future_track_utilization", "future_average_network_delay"
        ]
        
        all_cols = [c for c in df.columns if c not in exclude_cols]

        for col in all_cols:
            if df[col].dtype == "object":
                self.categorical_cols.append(col)
            elif df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                unique_vals = set(df[col].dropna().unique())
                if unique_vals.issubset({0, 1, 0.0, 1.0}):
                    self.boolean_cols.append(col)
                else:
                    self.numerical_cols.append(col)
            elif df[col].dtype == "bool":
                self.boolean_cols.append(col)

        # Impute missing values
        for col in self.numerical_cols:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
            
        for col in self.categorical_cols:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
            df[col] = df[col].fillna(mode_val)
            
        for col in self.boolean_cols:
            df[col] = df[col].fillna(0).astype(int)

        # -------------------------------------------------------------
        # Automatic Feature Selection Validation Checks
        # -------------------------------------------------------------
        # Encode categoricals temporarily to find correlation thresholds
        temp_encoded = pd.DataFrame(index=df.index)
        for col in self.categorical_cols:
            le = LabelEncoder()
            temp_encoded[col] = le.fit_transform(df[col].astype(str))

        temp_all = pd.concat([df[self.numerical_cols], temp_encoded, df[self.boolean_cols]], axis=1)

        # 1. Remove Constant Features (nunique <= 1)
        constant_features = [c for c in temp_all.columns if temp_all[c].nunique() <= 1]
        self.dropped_features.extend(constant_features)

        # 2. Remove Highly Correlated Features (Pearson Correlation > 0.98, skipping rolling/trend features)
        temp_non_const = temp_all.drop(columns=constant_features, errors="ignore")
        corr_matrix = temp_non_const.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        correlated_to_drop = []
        for column in upper.columns:
            # Skip correlation pruning for rolling history, trend variables, and core base columns
            if any(term in column for term in ["_roll", "_trend", "occupancy", "congestion"]):
                continue
            if any(upper[column] > 0.98):
                correlated_to_drop.append(column)
        self.dropped_features.extend(correlated_to_drop)

        train_logger.info(f"[{self.level_name.upper()}] Dropping constant features: {constant_features}")
        train_logger.info(f"[{self.level_name.upper()}] Dropping highly correlated features (>0.95): {correlated_to_drop}")

        # Filter feature lists before fitting estimators
        self.numerical_cols = [c for c in self.numerical_cols if c not in self.dropped_features]
        self.categorical_cols = [c for c in self.categorical_cols if c not in self.dropped_features]
        self.boolean_cols = [c for c in self.boolean_cols if c not in self.dropped_features]

        # Fit final encoders
        encoded_categoricals = pd.DataFrame(index=df.index)
        for col in self.categorical_cols:
            le = LabelEncoder()
            encoded_categoricals[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # Fit final Scaler (only on remaining numerical cols)
        scaled_numerical = pd.DataFrame(
            self.scaler.fit_transform(df[self.numerical_cols]),
            columns=self.numerical_cols,
            index=df.index
        )

        # Assemble final features
        X_processed = pd.concat([
            scaled_numerical,
            encoded_categoricals,
            df[self.boolean_cols]
        ], axis=1)

        self.feature_ordering = list(X_processed.columns)
        train_logger.info(f"[{self.level_name.upper()}] Remaining features after selection filter: {len(self.feature_ordering)}")
        return X_processed, df[target_cols]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms test or live inference rows.
        """
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        for col in self.numerical_cols:
            if col in df.columns:
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df[col] = df[col].fillna(median_val)
            else:
                df[col] = 0.0
            
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
                df[col] = df[col].fillna(mode_val)
            else:
                df[col] = "UNKNOWN"
            
        for col in self.boolean_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
            else:
                df[col] = 0

        # Transform categoricals (safely mapping unseen values)
        encoded_categoricals = pd.DataFrame(index=df.index)
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            known_classes = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            encoded_categoricals[col] = le.transform(df[col])

        # Scale numericals
        scaled_numerical = pd.DataFrame(
            self.scaler.transform(df[self.numerical_cols]),
            columns=self.numerical_cols,
            index=df.index
        )

        X_processed = pd.concat([
            scaled_numerical,
            encoded_categoricals,
            df[self.boolean_cols]
        ], axis=1)

        # reindex automatically drops columns not present in self.feature_ordering (including validation drops)
        X_processed = X_processed.reindex(columns=self.feature_ordering, fill_value=0)
        return X_processed

    def save_preprocessor(self, models_dir: str):
        """
        Saves scaler, encoders and metadata.
        """
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(models_dir, f"{self.level_name}_scaler.pkl"))
        joblib.dump(self.label_encoders, os.path.join(models_dir, f"{self.level_name}_encoders.pkl"))
        meta = {
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
            "boolean_cols": self.boolean_cols,
            "feature_ordering": self.feature_ordering,
            "dropped_features": self.dropped_features
        }
        joblib.dump(meta, os.path.join(models_dir, f"{self.level_name}_metadata.pkl"))
        train_logger.info(f"[{self.level_name.upper()}] Preprocessors saved successfully to {models_dir}")

    def load_preprocessor(self, models_dir: str):
        """
        Loads preprocessor files.
        """
        self.scaler = joblib.load(os.path.join(models_dir, f"{self.level_name}_scaler.pkl"))
        self.label_encoders = joblib.load(os.path.join(models_dir, f"{self.level_name}_encoders.pkl"))
        meta = joblib.load(os.path.join(models_dir, f"{self.level_name}_metadata.pkl"))
        self.numerical_cols = meta["numerical_cols"]
        self.categorical_cols = meta["categorical_cols"]
        self.boolean_cols = meta["boolean_cols"]
        self.feature_ordering = meta["feature_ordering"]
        self.dropped_features = meta.get("dropped_features", [])
        train_logger.info(f"[{self.level_name.upper()}] Preprocessors loaded successfully from {models_dir}")
