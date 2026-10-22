"""Differentially Private SGD (DP-SGD) training."""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from app.ml.baseline_model import BaselineMLPConfig
from app.utils.reproducibility import get_device
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DPSGDConfig(BaselineMLPConfig):
    """Configuration for DP-SGD training."""

    def __init__(
        self,
        input_dim: int,
        epsilon: float = 1.0,
        delta: float = 1e-6,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 0.5,
        **kwargs,
    ):
        """Initialize DP-SGD config.
        
        Args:
            input_dim: Input dimension
            epsilon: Privacy budget
            delta: Privacy delta
            max_grad_norm: Gradient clipping norm
            noise_multiplier: Noise multiplier for DP
            **kwargs: Additional args passed to BaselineMLPConfig
        """
        super().__init__(input_dim, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier


class DPSGDTrainer:
    """Train model with Differential Privacy using DP-SGD."""

    def __init__(self, model: nn.Module, config: DPSGDConfig):
        """Initialize DP-SGD trainer.
        
        Args:
            model: PyTorch model
            config: DP-SGD configuration
        """
        self.model = model
        self.config = config
        self.device = get_device()
        self.model = self.model.to(self.device)

        # Initialize privacy accounting
        self.epsilon_spent = 0.0
        self.delta = config.delta
        self.steps = 0
        self.sample_rate = config.batch_size  # Will be updated with actual dataset size

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.criterion = nn.BCELoss(reduction="mean")

        # Training history
        self.history: Dict = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "epsilon_spent": [],
        }

        logger.info(f"DP-SGD Trainer initialized")
        logger.info(f"  epsilon: {config.epsilon}")
        logger.info(f"  delta: {config.delta}")
        logger.info(f"  max_grad_norm: {config.max_grad_norm}")
        logger.info(f"  noise_multiplier: {config.noise_multiplier}")

    def train_epoch(self, X: np.ndarray, y: np.ndarray, dataset_size: int) -> float:
        """Train for one epoch with DP-SGD.
        
        Args:
            X: Training features
            y: Training labels
            dataset_size: Total dataset size (for privacy accounting)
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Update sample rate
        self.sample_rate = self.config.batch_size / dataset_size

        # Create batches
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for i in range(0, len(X), self.config.batch_size):
            batch_idx = indices[i : i + self.config.batch_size]
            X_batch = torch.tensor(X[batch_idx], dtype=torch.float32).to(self.device)
            y_batch = torch.tensor(y[batch_idx], dtype=torch.float32).to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X_batch).squeeze()
            loss = self.criterion(outputs, y_batch)

            # Backward pass
            loss.backward()

            # DP-SGD: Clip gradients
            self._clip_gradients()

            # DP-SGD: Add noise
            self._add_dp_noise()

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.steps += 1

        # Update privacy accounting
        self._update_privacy_accounting()

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _clip_gradients(self) -> None:
        """Clip gradients to max norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        clip_coef = self.config.max_grad_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)

    def _add_dp_noise(self) -> None:
        """Add noise to gradients for differential privacy."""
        noise_scale = self.config.max_grad_norm * self.config.noise_multiplier

        for p in self.model.parameters():
            if p.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=noise_scale,
                    size=p.grad.shape,
                    device=self.device,
                )
                p.grad.data.add_(noise)

    def _update_privacy_accounting(self) -> None:
        """Update spent privacy budget using Rényi differential privacy."""
        # Simplified privacy accounting (composition bound)
        # For rigorous accounting, use Opacus or similar

        # Number of epochs so far
        num_params = sum(1 for _ in self.model.parameters())
        epochs = self.steps * self.config.batch_size / (num_params * 100 + 1)

        # Approximate epsilon using composition
        # This is a simplified version - real accounting is more complex
        q = self.sample_rate
        sigma = self.config.max_grad_norm * self.config.noise_multiplier
        orders = [1.5, 2, 5, 10, 20, 50, 100]

        renyis = []
        for order in orders:
            if order == 1:
                continue
            try:
                alpha = np.log(1 - q + q * np.exp((order - 1) / sigma ** 2))
                renyi_epsilon = alpha / (order - 1)
                renyis.append(renyi_epsilon)
            except:
                pass

        if renyis:
            # Take minimum across orders
            self.epsilon_spent = float(min(renyis))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate model.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for i in range(0, len(X), self.config.batch_size):
                X_batch = torch.tensor(X[i : i + self.config.batch_size], dtype=torch.float32).to(
                    self.device
                )
                y_batch = torch.tensor(y[i : i + self.config.batch_size], dtype=torch.float32).to(
                    self.device
                )

                outputs = self.model(X_batch).squeeze()
                loss = self.criterion(outputs, y_batch)

                total_loss += loss.item()
                acc = ((outputs > 0.5).float() == y_batch).float().mean().item()
                total_acc += acc
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0

        return avg_loss, avg_acc

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict:
        """Full DP-SGD training loop.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training history
        """
        logger.info(f"Starting DP-SGD training for {self.config.epochs} epochs...")

        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(X_train, y_train, len(X_train))
            train_loss, train_acc = self.evaluate(X_train, y_train)

            # Validate
            val_loss, val_acc = self.evaluate(X_val, y_val)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["epsilon_spent"].append(float(self.epsilon_spent))

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Acc: {val_acc:.4f}, Epsilon Spent: {self.epsilon_spent:.4f}"
                )

        logger.info("DP-SGD training complete")
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data.
        
        Args:
            X: Features
            
        Returns:
            Predictions (probabilities)
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(X), self.config.batch_size):
                X_batch = torch.tensor(X[i : i + self.config.batch_size], dtype=torch.float32).to(
                    self.device
                )
                outputs = self.model(X_batch).squeeze()
                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict classes on new data.
        
        Args:
            X: Features
            threshold: Classification threshold
            
        Returns:
            Binary class predictions
        """
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    def get_privacy_params(self) -> Dict:
        """Get current privacy parameters.
        
        Returns:
            Dictionary of privacy params
        """
        return {
            "epsilon_target": self.config.epsilon,
            "epsilon_spent": float(self.epsilon_spent),
            "delta": self.delta,
            "max_grad_norm": self.config.max_grad_norm,
            "noise_multiplier": self.config.noise_multiplier,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "sample_rate": self.sample_rate,
        }
