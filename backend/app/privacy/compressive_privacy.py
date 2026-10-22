"""Compressive Privacy using PCA."""

import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Optional
from app.utils.logging import get_logger

logger = get_logger(__name__)


class CompressivePrivacy:
    """Apply compressive privacy to feature representations."""

    def __init__(self, compression_ratio: float = 0.8, random_state: int = 42):
        """Initialize compressive privacy.
        
        Args:
            compression_ratio: Target ratio of compressed dimensions to original
                             (0.8 means keep 80% of variance, reduce to ~80% features)
            random_state: Random state for reproducibility
        """
        self.compression_ratio = compression_ratio
        self.random_state = random_state
        self.pca: Optional[PCA] = None
        self.original_dim: Optional[int] = None
        self.compressed_dim: Optional[int] = None
        logger.info(f"CompressivePrivacy initialized with ratio={compression_ratio}")

    def fit(self, X: np.ndarray) -> "CompressivePrivacy":
        """Fit PCA on data.
        
        Args:
            X: Training features (num_samples, num_features)
            
        Returns:
            Self for chaining
        """
        self.original_dim = X.shape[1]
        
        # Determine number of components to keep based on compression ratio
        # compression_ratio of 0.8 means keep 80% of variance
        self.pca = PCA(n_components=self.compression_ratio, random_state=self.random_state)
        self.pca.fit(X)
        
        self.compressed_dim = self.pca.n_components_
        
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        logger.info(
            f"Fitted PCA: {self.original_dim} -> {self.compressed_dim} dimensions, "
            f"explained variance: {explained_var:.4f}"
        )
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compress features using fitted PCA.
        
        Args:
            X: Features to compress
            
        Returns:
            Compressed features
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            X: Features
            
        Returns:
            Compressed features
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_compressed: np.ndarray) -> np.ndarray:
        """Reconstruct features from compressed representation.
        
        Args:
            X_compressed: Compressed features
            
        Returns:
            Reconstructed features
        """
        if self.pca is None:
            raise ValueError("PCA not fitted.")
        
        return self.pca.inverse_transform(X_compressed)

    def get_compression_stats(self) -> dict:
        """Get compression statistics.
        
        Returns:
            Dictionary of stats
        """
        if self.pca is None:
            return {}
        
        return {
            "original_dim": self.original_dim,
            "compressed_dim": self.compressed_dim,
            "compression_ratio": self.compressed_dim / self.original_dim if self.original_dim else 0,
            "explained_variance": float(np.sum(self.pca.explained_variance_ratio_)),
            "variance_per_component": self.pca.explained_variance_ratio_.tolist(),
        }
