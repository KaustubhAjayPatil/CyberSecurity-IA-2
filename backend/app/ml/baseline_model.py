"""Baseline MLP classifier for fraud detection."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from app.utils.reproducibility import get_device
from app.utils.logging import get_logger

logger = get_logger(__name__)


class BaselineMLPConfig:
    """Configuration for Baseline MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[list] = None,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
    ):
        """Initialize config.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Explicit list of hidden layer dimensions
            hidden_dim: Hidden layer dimension (if hidden_dims not provided)
            num_hidden_layers: Number of hidden layers
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [hidden_dim] * num_hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs


class BaselineMLP(nn.Module):
    """Baseline MLP for fraud classification."""

    def __init__(self, config: BaselineMLPConfig):
        """Initialize MLP model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Build layers
        layers = []
        prev_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output probability (0-1)
        """
        return self.model(x)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without sigmoid (for internal use).
        
        Args:
            x: Input tensor
            
        Returns:
            Raw output
        """
        return self.model[:-1](x)  # Everything except final sigmoid


class BaselineTrainer:
    """Train and evaluate baseline MLP."""

    def __init__(self, model: BaselineMLP, config: BaselineMLPConfig):
        """Initialize trainer.
        
        Args:
            model: MLP model
            config: Model configuration
        """
        self.model = model
        self.config = config
        self.device = get_device()
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.criterion = nn.BCELoss()
        self.history: Dict = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        logger.info(f"Using device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train for one epoch.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Create batches
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for i in range(0, len(X), self.config.batch_size):
            batch_idx = indices[i : i + self.config.batch_size]
            X_batch = torch.tensor(X[batch_idx], dtype=torch.float32).to(self.device)
            y_batch = torch.tensor(y[batch_idx], dtype=torch.float32).to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(X_batch).squeeze()
            loss = self.criterion(outputs, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

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

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Full training loop.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training history
        """
        logger.info(f"Training for {self.config.epochs} epochs...")

        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(X_train, y_train)
            train_loss, train_acc = self.evaluate(X_train, y_train)

            # Validate
            val_loss, val_acc = self.evaluate(X_val, y_val)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
                )

        logger.info("Training complete")
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
            Predicted classes (0 or 1)
        """
        probs = self.predict(X)
        return (probs > threshold).astype(int)
