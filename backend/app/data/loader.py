"""Dataset loading and validation."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetLoader:
    """Load and validate financial transaction dataset."""

    REQUIRED_TARGET = "fraud_bool"

    def __init__(self, dataset_path: str, sample_size: Optional[int] = None):
        """Initialize dataset loader.
        
        Args:
            dataset_path: Path to CSV file
            sample_size: Optional number of samples to use
        """
        self.dataset_path = Path(dataset_path)
        self.sample_size = sample_size
        self._validate_path()

    def _validate_path(self) -> None:
        """Validate that dataset file exists."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        if self.dataset_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected CSV file, got: {self.dataset_path.suffix}")

    def load(self) -> pd.DataFrame:
        """Load dataset from file.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            ValueError: If dataset is invalid
        """
        logger.info(f"Loading dataset from {self.dataset_path}")

        try:
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")

        # Validate target column
        if self.REQUIRED_TARGET not in df.columns:
            raise ValueError(
                f"Target column '{self.REQUIRED_TARGET}' not found in dataset. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Sample if requested
        if self.sample_size is not None:
            if self.sample_size > len(df):
                logger.warning(
                    f"Sample size {self.sample_size} larger than dataset {len(df)}, "
                    f"using full dataset"
                )
            else:
                df = df.sample(n=self.sample_size, random_state=42)
                logger.info(f"Sampled to {len(df)} rows")

        return df

    def validate(self, df: pd.DataFrame) -> dict:
        """Validate dataset integrity.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation report dictionary
        """
        report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {},
        }

        # Check target
        if self.REQUIRED_TARGET not in df.columns:
            report["is_valid"] = False
            report["errors"].append(f"Missing target column: {self.REQUIRED_TARGET}")
            return report

        # Check for empty dataset
        if len(df) == 0:
            report["is_valid"] = False
            report["errors"].append("Dataset is empty")
            return report

        # Check features
        if len(df.columns) < 2:
            report["is_valid"] = False
            report["errors"].append(f"Dataset has only {len(df.columns)} column(s), need at least 2")

        # Check for all missing columns
        missing_cols = df.columns[df.isnull().all()].tolist()
        if missing_cols:
            report["warnings"].append(f"Columns with all missing values: {missing_cols}")

        # Statistics
        report["statistics"] = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "num_missing": int(df.isnull().sum().sum()),
            "missing_percentage": float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
            "target_distribution": df[self.REQUIRED_TARGET].value_counts().to_dict(),
            "numerical_features": df.select_dtypes(include=["number"]).columns.tolist(),
            "categorical_features": df.select_dtypes(include=["object"]).columns.tolist(),
        }

        return report
