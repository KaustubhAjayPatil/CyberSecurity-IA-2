"""Dataset profiling and analysis."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetProfiler:
    """Analyze dataset structure and characteristics."""

    # Keywords suggesting sensitive attributes
    SENSITIVE_KEYWORDS = {
        "age": ["age", "born", "dob"],
        "employment": ["employment", "job", "occupation", "profession"],
        "housing": ["housing", "house", "home", "residence"],
        "income": ["income", "salary", "wage", "earnings"],
        "credit": ["credit", "score", "risk", "rating"],
        "phone": ["phone", "mobile", "contact"],
        "email": ["email", "mail"],
    }

    # Columns that are likely direct identifiers (should be removed)
    IDENTIFIER_KEYWORDS = [
        "id",
        "customer_id",
        "account_id",
        "transaction_id",
        "name",
        "ssn",
        "passport",
        "address",
    ]

    def __init__(self, df: pd.DataFrame):
        """Initialize profiler with dataset.
        
        Args:
            df: DataFrame to profile
        """
        self.df = df
        self.profile: Dict[str, Any] = {}

    def profile_dataset(self) -> Dict[str, Any]:
        """Generate comprehensive dataset profile.
        
        Returns:
            Profile dictionary
        """
        logger.info("Profiling dataset...")

        self.profile = {
            "shape": self.df.shape,
            "columns": len(self.df.columns),
            "rows": len(self.df),
            "column_info": self._analyze_columns(),
            "missing_values": self._analyze_missing(),
            "feature_types": self._categorize_features(),
            "sensitive_attributes": self._recommend_sensitive_attributes(),
            "identifiers_to_remove": self._identify_direct_identifiers(),
            "target_distribution": self._analyze_target(),
            "class_imbalance": self._calculate_imbalance(),
            "cardinality": self._analyze_cardinality(),
        }

        return self.profile

    def _analyze_columns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze individual columns.
        
        Returns:
            Dictionary of column analysis
        """
        info = {}
        for col in self.df.columns:
            info[col] = {
                "dtype": str(self.df[col].dtype),
                "non_null_count": int(self.df[col].notna().sum()),
                "null_count": int(self.df[col].isna().sum()),
                "unique_values": int(self.df[col].nunique()),
                "is_numeric": pd.api.types.is_numeric_dtype(self.df[col]),
                "is_categorical": pd.api.types.is_object_dtype(self.df[col]),
            }

            if info[col]["is_numeric"]:
                info[col].update({
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max()),
                    "mean": float(self.df[col].mean()),
                    "std": float(self.df[col].std()),
                })

            if info[col]["is_categorical"]:
                info[col]["unique_categories"] = self.df[col].unique()[:10].tolist()

        return info

    def _analyze_missing(self) -> Dict[str, float]:
        """Analyze missing value patterns.
        
        Returns:
            Missing value percentages by column
        """
        missing = {}
        for col in self.df.columns:
            missing[col] = float(self.df[col].isna().sum() / len(self.df) * 100)
        return missing

    def _categorize_features(self) -> Dict[str, List[str]]:
        """Categorize features by type.
        
        Returns:
            Dictionary of feature categories
        """
        return {
            "numerical": self.df.select_dtypes(include=["number"]).columns.tolist(),
            "categorical": self.df.select_dtypes(include=["object"]).columns.tolist(),
            "boolean": self.df.select_dtypes(include=["bool"]).columns.tolist(),
        }

    def _recommend_sensitive_attributes(self) -> Dict[str, List[str]]:
        """Recommend sensitive attributes based on feature names and types.
        
        Returns:
            Dictionary of recommended sensitive attributes by category
        """
        recommendations = {}

        for category, keywords in self.SENSITIVE_KEYWORDS.items():
            matches = []
            for col in self.df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in keywords):
                    matches.append({
                        "column": col,
                        "dtype": str(self.df[col].dtype),
                        "unique_values": int(self.df[col].nunique()),
                        "is_numeric": pd.api.types.is_numeric_dtype(self.df[col]),
                    })
            if matches:
                recommendations[category] = matches

        return recommendations

    def _identify_direct_identifiers(self) -> List[str]:
        """Identify columns that are likely direct identifiers.
        
        Returns:
            List of suspected identifier columns
        """
        identifiers = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in self.IDENTIFIER_KEYWORDS):
                identifiers.append(col)
            # High cardinality might also indicate identifier
            if self.df[col].nunique() == len(self.df):
                if col_lower not in ["fraud_bool", "target"]:
                    identifiers.append(col)

        return list(set(identifiers))

    def _analyze_target(self) -> Dict[str, Any]:
        """Analyze target variable distribution.
        
        Returns:
            Target analysis
        """
        if "fraud_bool" not in self.df.columns:
            return {}

        target = self.df["fraud_bool"]
        value_counts = target.value_counts()

        return {
            "value_counts": value_counts.to_dict(),
            "percentages": (value_counts / len(target) * 100).to_dict(),
            "null_count": int(target.isna().sum()),
        }

    def _calculate_imbalance(self) -> Dict[str, float]:
        """Calculate class imbalance ratio.
        
        Returns:
            Imbalance metrics
        """
        if "fraud_bool" not in self.df.columns:
            return {}

        target = self.df["fraud_bool"]
        value_counts = target.value_counts()

        if len(value_counts) != 2:
            return {}

        minority = value_counts.min()
        majority = value_counts.max()

        return {
            "majority_class": int(majority),
            "minority_class": int(minority),
            "imbalance_ratio": float(majority / minority),
            "minority_percentage": float(minority / len(target) * 100),
        }

    def _analyze_cardinality(self) -> Dict[str, List[str]]:
        """Identify high and low cardinality columns.
        
        Returns:
            Cardinality analysis
        """
        high_cardinality = []
        low_cardinality = []

        for col in self.df.columns:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio > 0.5:
                high_cardinality.append(col)
            elif self.df[col].nunique() <= 5:
                low_cardinality.append(col)

        return {
            "high_cardinality": high_cardinality,
            "low_cardinality": low_cardinality,
        }

    def generate_report(self) -> str:
        """Generate human-readable profile report.
        
        Returns:
            Formatted report string
        """
        if not self.profile:
            self.profile_dataset()

        report = []
        report.append("=" * 80)
        report.append("DATASET PROFILE REPORT")
        report.append("=" * 80)

        report.append(f"\nShape: {self.profile['rows']} rows × {self.profile['columns']} columns")

        report.append("\n--- Feature Types ---")
        for ftype, features in self.profile["feature_types"].items():
            report.append(f"{ftype}: {len(features)} columns")

        report.append("\n--- Recommended Sensitive Attributes ---")
        if self.profile["sensitive_attributes"]:
            for category, attrs in self.profile["sensitive_attributes"].items():
                report.append(f"\n{category.upper()}:")
                for attr in attrs:
                    report.append(
                        f"  - {attr['column']} ({attr['dtype']}, "
                        f"{attr['unique_values']} unique values)"
                    )
        else:
            report.append("No sensitive attributes detected")

        report.append("\n--- Direct Identifiers to Remove ---")
        if self.profile["identifiers_to_remove"]:
            for col in self.profile["identifiers_to_remove"]:
                report.append(f"  - {col}")
        else:
            report.append("No direct identifiers detected")

        report.append("\n--- Target Distribution (fraud_bool) ---")
        if self.profile["target_distribution"]:
            dist = self.profile["target_distribution"]["value_counts"]
            for value, count in dist.items():
                pct = dist[value] / sum(dist.values()) * 100
                report.append(f"  {value}: {count:,} ({pct:.2f}%)")

            if self.profile["class_imbalance"]:
                imbalance = self.profile["class_imbalance"]
                report.append(f"\n  Imbalance Ratio: {imbalance['imbalance_ratio']:.2f}:1")
                report.append(f"  Minority Class %: {imbalance['minority_percentage']:.2f}%")

        report.append("\n" + "=" * 80)

        return "\n".join(report)
