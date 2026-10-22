"""Input privacy modification - removing identifiers and generalizing sensitive attributes."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from app.utils.logging import get_logger

logger = get_logger(__name__)


class InputPrivacyModification:
    """Modify input data to protect privacy of sensitive attributes."""

    def __init__(self):
        """Initialize input modification."""
        self.identifiers_removed: List[str] = []
        self.sensitive_attributes_generalized: Dict[str, dict] = {}
        logger.info("InputPrivacyModification initialized")

    def remove_identifiers(
        self,
        df: pd.DataFrame,
        identifiers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Remove direct identifiers from dataset.
        
        Args:
            df: Input dataframe
            identifiers: List of column names to remove
            
        Returns:
            Dataframe with identifiers removed
        """
        if identifiers is None:
            identifiers = []

        result = df.copy()
        removed = []

        for col in identifiers:
            if col in result.columns:
                result = result.drop(columns=[col])
                removed.append(col)
                logger.info(f"Removed identifier: {col}")

        self.identifiers_removed = removed
        return result

    def generalize_age(
        self,
        df: pd.DataFrame,
        age_column: str = "customer_age",
        bin_width: int = 10,
    ) -> pd.DataFrame:
        """Generalize age into age groups.
        
        Args:
            df: Input dataframe
            age_column: Name of age column
            bin_width: Width of age bins (e.g., 10 for 0-9, 10-19, etc.)
            
        Returns:
            Dataframe with generalized age
        """
        result = df.copy()

        if age_column not in result.columns:
            return result

        # Create age groups
        def age_to_group(age):
            lower = (int(age) // bin_width) * bin_width
            upper = lower + bin_width - 1
            return f"{lower}-{upper}"

        result[age_column] = result[age_column].apply(age_to_group)

        self.sensitive_attributes_generalized[age_column] = {
            "method": "binning",
            "bin_width": bin_width,
        }

        logger.info(f"Generalized {age_column} into age groups (bin_width={bin_width})")
        return result

    def generalize_numerical(
        self,
        df: pd.DataFrame,
        column: str,
        num_bins: int = 5,
    ) -> pd.DataFrame:
        """Generalize numerical attribute into bins.
        
        Args:
            df: Input dataframe
            column: Column name
            num_bins: Number of bins
            
        Returns:
            Dataframe with generalized column
        """
        result = df.copy()

        if column not in result.columns:
            return result

        # Quantile-based binning
        result[column] = pd.qcut(result[column], q=num_bins, labels=False, duplicates="drop")

        self.sensitive_attributes_generalized[column] = {
            "method": "quantile_binning",
            "num_bins": num_bins,
        }

        logger.info(f"Generalized {column} into {num_bins} quantile bins")
        return result

    def suppress_high_cardinality(
        self,
        df: pd.DataFrame,
        cardinality_threshold: int = 100,
    ) -> pd.DataFrame:
        """Remove or generalize high-cardinality categorical columns.
        
        Args:
            df: Input dataframe
            cardinality_threshold: Cardinality threshold
            
        Returns:
            Dataframe with high-cardinality columns removed
        """
        result = df.copy()
        removed = []

        for col in result.select_dtypes(include=["object"]).columns:
            if result[col].nunique() > cardinality_threshold:
                result = result.drop(columns=[col])
                removed.append(col)
                logger.info(f"Removed high-cardinality column: {col} ({result[col].nunique()} unique values)")

        return result

    def perturb_categorical(
        self,
        df: pd.DataFrame,
        column: str,
        perturbation_prob: float = 0.1,
    ) -> pd.DataFrame:
        """Add perturbation to categorical attribute values.
        
        Args:
            df: Input dataframe
            column: Column name
            perturbation_prob: Probability of changing value
            
        Returns:
            Dataframe with perturbed values
        """
        result = df.copy()

        if column not in result.columns:
            return result

        categories = result[column].unique()

        def perturb(val):
            if np.random.random() < perturbation_prob:
                return np.random.choice(categories)
            return val

        result[column] = result[column].apply(perturb)

        logger.info(f"Applied perturbation to {column} (prob={perturbation_prob})")
        return result

    def apply_modifications(
        self,
        df: pd.DataFrame,
        identifiers_to_remove: Optional[List[str]] = None,
        sensitive_attributes_to_transform: Optional[Dict[str, dict]] = None,
    ) -> pd.DataFrame:
        """Apply all privacy modifications.
        
        Args:
            df: Input dataframe
            identifiers_to_remove: List of identifier columns to remove
            sensitive_attributes_to_transform: Dict mapping column name to transformation params
                Example: {"customer_age": {"method": "bin", "bin_width": 10}}
            
        Returns:
            Privacy-modified dataframe
        """
        result = df.copy()

        # Remove identifiers
        if identifiers_to_remove:
            result = self.remove_identifiers(result, identifiers_to_remove)

        # Transform sensitive attributes
        if sensitive_attributes_to_transform:
            for col, params in sensitive_attributes_to_transform.items():
                method = params.get("method", "").lower()

                if method == "bin":
                    result = self.generalize_age(
                        result,
                        age_column=col,
                        bin_width=params.get("bin_width", 10),
                    )
                elif method == "quantile_bin":
                    result = self.generalize_numerical(
                        result,
                        column=col,
                        num_bins=params.get("num_bins", 5),
                    )
                elif method == "perturb":
                    result = self.perturb_categorical(
                        result,
                        column=col,
                        perturbation_prob=params.get("prob", 0.1),
                    )

        logger.info(f"Privacy modifications applied. Shape before: {df.shape}, after: {result.shape}")
        return result

    def get_modifications_summary(self) -> dict:
        """Get summary of applied modifications.
        
        Returns:
            Dictionary of applied modifications
        """
        return {
            "identifiers_removed": self.identifiers_removed,
            "sensitive_attributes_generalized": self.sensitive_attributes_generalized,
        }
