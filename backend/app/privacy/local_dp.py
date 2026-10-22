"""Local Differential Privacy (LDP) mechanisms for sensitive attributes."""

import numpy as np
from typing import List, Optional, Tuple
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LocalDifferentialPrivacy:
    """Apply Local Differential Privacy to selected attributes."""

    def __init__(self, epsilon: float = 1.0):
        """Initialize LDP mechanism.
        
        Args:
            epsilon: Privacy budget (smaller = stronger privacy)
        """
        self.epsilon = epsilon
        self.p = 1.0 / (1.0 + np.exp(epsilon))  # Randomization parameter
        logger.info(f"LDP initialized with epsilon={epsilon}, p={self.p:.4f}")

    def randomized_response_categorical(
        self,
        value: str,
        categories: Optional[List[str]] = None,
    ) -> str:
        """Apply randomized response to categorical value (single instance).
        
        Args:
            value: Original categorical value
            categories: List of all possible categories
            
        Returns:
            Randomized value
        """
        if categories is None:
            categories = []

        # With probability p, keep original; otherwise, pick random category
        if np.random.random() < self.p:
            return value
        else:
            if categories:
                return np.random.choice(categories)
            return value

    def randomized_response_array(
        self,
        values: np.ndarray,
        categories: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Apply randomized response to array of categorical values.
        
        Args:
            values: Array of categorical values
            categories: List of all possible categories
            
        Returns:
            Array of randomized values
        """
        if categories is None:
            categories = list(np.unique(values))

        randomized = []
        for val in values:
            randomized.append(self.randomized_response_categorical(val, categories))

        return np.array(randomized)

    def laplace_noise_numerical(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Laplace noise to numerical value.
        
        Args:
            value: Original numerical value
            sensitivity: Sensitivity (max change from one record change)
            
        Returns:
            Noisy value
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(loc=0, scale=scale)
        return value + noise

    def laplace_noise_array(
        self,
        values: np.ndarray,
        sensitivity: float = 1.0,
        normalize: bool = False,
    ) -> np.ndarray:
        """Add Laplace noise to array of numerical values.
        
        Args:
            values: Array of numerical values
            sensitivity: Sensitivity
            normalize: Whether to normalize back to original range
            
        Returns:
            Array of noisy values
        """
        original_min = values.min()
        original_max = values.max()
        original_range = original_max - original_min if original_max > original_min else 1.0

        scale = sensitivity / self.epsilon
        noise = np.random.laplace(loc=0, scale=scale, size=len(values))
        noisy = values + noise

        if normalize:
            # Clip to original range
            noisy = np.clip(noisy, original_min, original_max)

        return noisy
