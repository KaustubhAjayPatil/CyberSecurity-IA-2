"""Data preprocessing pipeline."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline:
    """Handle data preprocessing with proper train/test separation."""

    def __init__(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_seed: int = 42,
        target_column: str = "fraud_bool",
        remove_columns: Optional[List[str]] = None,
    ):
        """Initialize preprocessing pipeline.
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_seed: Random seed for reproducibility
            target_column: Name of target column
            remove_columns: Columns to remove
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed
        self.target_column = target_column
        self.remove_columns = remove_columns or []

        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: Optional[List[str]] = None
        self.categorical_features: List[str] = []
        self.numerical_features: List[str] = []

    def fit_and_transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit preprocessing on full dataset and return train/val/test split.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Starting preprocessing pipeline...")

        # Remove target and specified columns
        X = df.drop(columns=[self.target_column] + self.remove_columns, errors="ignore")
        y = df[self.target_column].values

        logger.info(f"Features shape before preprocessing: {X.shape}")
        logger.info(f"Target distribution: {np.unique(y, return_counts=True)}")

        # Identify categorical and numerical features
        self.categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
        self.numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

        logger.info(f"Categorical features: {self.categorical_features}")
        logger.info(f"Numerical features: {self.numerical_features}")

        # First split: train (70%) and temp (30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=self.test_size + self.val_size,
            random_state=self.random_seed,
            stratify=y,
        )

        # Second split: val (15%) and test (15%)
        val_ratio = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1 - val_ratio,
            random_state=self.random_seed,
            stratify=y_temp,
        )

        logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
        logger.info(f"Target distribution - Train: {np.unique(y_train, return_counts=True)}")

        # Encode categorical features (fit only on train)
        X_train = self._encode_categorical(X_train, fit=True)
        X_val = self._encode_categorical(X_val, fit=False)
        X_test = self._encode_categorical(X_test, fit=False)

        # Scale numerical features (fit only on train)
        X_train = self._scale_numerical(X_train, fit=True)
        X_val = self._scale_numerical(X_val, fit=False)
        X_test = self._scale_numerical(X_test, fit=False)

        # Store feature names
        self.feature_names = X_train.columns.tolist()

        logger.info(f"Final feature count: {len(self.feature_names)}")
        logger.info("Preprocessing complete")

        return (
            X_train.values,
            X_val.values,
            X_test.values,
            y_train,
            y_val,
            y_test,
        )

    def _encode_categorical(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            X: Input dataframe
            fit: Whether to fit the encoders
            
        Returns:
            Dataframe with encoded categorical features
        """
        X_encoded = X.copy()

        for col in self.categorical_features:
            if col not in X_encoded.columns:
                continue

            if fit:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    X_encoded[col] = le.transform(X_encoded[col].astype(str))

        return X_encoded

    def _scale_numerical(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale numerical features using StandardScaler.
        
        Args:
            X: Input dataframe
            fit: Whether to fit the scaler
            
        Returns:
            Dataframe with scaled numerical features
        """
        X_scaled = X.copy()

        if fit:
            self.scaler = StandardScaler()
            X_scaled[self.numerical_features] = self.scaler.fit_transform(
                X_scaled[self.numerical_features]
            )
        else:
            if self.scaler:
                X_scaled[self.numerical_features] = self.scaler.transform(
                    X_scaled[self.numerical_features]
                )

        return X_scaled

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessing.
        
        Args:
            X: Input dataframe
            
        Returns:
            Transformed numpy array
        """
        X_transformed = X.copy()
        X_transformed = self._encode_categorical(X_transformed, fit=False)
        X_transformed = self._scale_numerical(X_transformed, fit=False)
        return X_transformed.values

    def inverse_transform_numerical(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled numerical features.
        
        Args:
            X: Scaled numpy array
            
        Returns:
            Unscaled numpy array
        """
        if self.scaler:
            X_unscaled = X.copy()
            # Only inverse transform numerical feature columns
            numerical_indices = [
                i
                for i, col in enumerate(self.feature_names)
                if col in self.numerical_features
            ]
            if numerical_indices:
                X_unscaled[:, numerical_indices] = self.scaler.inverse_transform(
                    X[:, numerical_indices]
                )
            return X_unscaled
        return X

    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing.
        
        Returns:
            List of feature names
        """
        return self.feature_names or []

    def get_input_dim(self) -> int:
        """Get input dimension for ML models.
        
        Returns:
            Number of features
        """
        return len(self.feature_names) if self.feature_names else 0
