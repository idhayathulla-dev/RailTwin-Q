import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ai.delay_prediction.utils import train_logger

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []
        self.boolean_cols = []

    def fit_transform(self, df: pd.DataFrame, target_cols: list) -> tuple:
        """
        Learns preprocessing transformations on training df and returns scaled/encoded data.
        """
        train_logger.info("Fitting and transforming preprocessors...")
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Identify column types automatically (excluding target leakage arrival times and identifiers)
        exclude_cols = target_cols + [
            "scenario_id", "tick", "state_id", "time", "train_name", 
            "current_station", "next_station", "expected_arrival_time", 
            "scheduled_arrival_time", "active_disruptions",
            "train_no", "route_id", "current_station_id", "next_station_id", 
            "current_track_id"
        ]
        all_cols = [c for c in df.columns if c not in exclude_cols]
        
        for col in all_cols:
            if df[col].dtype == "object":
                self.categorical_cols.append(col)
            elif df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                # If unique values are just 0 and 1, it's boolean
                unique_vals = set(df[col].dropna().unique())
                if unique_vals.issubset({0, 1, 0.0, 1.0}):
                    self.boolean_cols.append(col)
                else:
                    self.numerical_cols.append(col)
            elif df[col].dtype == "bool":
                self.boolean_cols.append(col)

        train_logger.info(f"Feature types detected: Numerical={len(self.numerical_cols)}, Categorical={len(self.categorical_cols)}, Boolean={len(self.boolean_cols)}")

        # Impute missing values (median for numerical, mode for categoricals)
        for col in self.numerical_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
        for col in self.categorical_cols:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
            df[col] = df[col].fillna(mode_val)
            
        for col in self.boolean_cols:
            df[col] = df[col].fillna(0).astype(int)

        # Fit & Transform Categorical Encoders
        encoded_categoricals = pd.DataFrame(index=df.index)
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Convert series to string to be safe
            df[col] = df[col].astype(str)
            encoded_categoricals[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # Fit & Transform Scaler
        scaled_numerical = pd.DataFrame(
            self.scaler.fit_transform(df[self.numerical_cols]),
            columns=self.numerical_cols,
            index=df.index
        )

        # Assemble final processed features
        X_processed = pd.concat([
            scaled_numerical,
            encoded_categoricals,
            df[self.boolean_cols]
        ], axis=1)

        # Save feature ordering for transform consistency
        self.feature_ordering = list(X_processed.columns)

        train_logger.info(f"Preprocessing fit complete. Features shape: {X_processed.shape}")
        return X_processed, df[target_cols]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms test or live inference df using already fit preprocessing variables.
        """
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Impute missing values
        for col in self.numerical_cols:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                df[col] = 0.0  # Fallback
            
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

        # Transform categoricals (handling unseen labels safely)
        encoded_categoricals = pd.DataFrame(index=df.index)
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            # Map unseen labels to first label in encoder class list
            known_classes = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            encoded_categoricals[col] = le.transform(df[col])

        # Scale numericals
        scaled_numerical = pd.DataFrame(
            self.scaler.transform(df[self.numerical_cols]),
            columns=self.numerical_cols,
            index=df.index
        )

        # Assemble features
        X_processed = pd.concat([
            scaled_numerical,
            encoded_categoricals,
            df[self.boolean_cols]
        ], axis=1)

        # Reindex to ensure same feature ordering as fit
        X_processed = X_processed.reindex(columns=self.feature_ordering, fill_value=0)
        return X_processed

    def save_preprocessors(self, models_dir: str):
        """
        Saves scaler, encoders, and schemas to disk.
        """
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(models_dir, "feature_scaler.pkl"))
        joblib.dump(self.label_encoders, os.path.join(models_dir, "label_encoders.pkl"))
        # Save columns config
        meta = {
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
            "boolean_cols": self.boolean_cols,
            "feature_ordering": self.feature_ordering
        }
        joblib.dump(meta, os.path.join(models_dir, "preprocessor_metadata.pkl"))
        train_logger.info(f"Preprocessors saved successfully to {models_dir}")

    def load_preprocessors(self, models_dir: str):
        """
        Loads preprocessor files from disk.
        """
        self.scaler = joblib.load(os.path.join(models_dir, "feature_scaler.pkl"))
        self.label_encoders = joblib.load(os.path.join(models_dir, "label_encoders.pkl"))
        meta = joblib.load(os.path.join(models_dir, "preprocessor_metadata.pkl"))
        self.numerical_cols = meta["numerical_cols"]
        self.categorical_cols = meta["categorical_cols"]
        self.boolean_cols = meta["boolean_cols"]
        self.feature_ordering = meta["feature_ordering"]
        train_logger.info(f"Preprocessors loaded successfully from {models_dir}")
