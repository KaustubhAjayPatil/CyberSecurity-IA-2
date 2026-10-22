"""Model Inversion Attack (MIA) - infer sensitive attributes from model predictions."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from app.utils.logging import get_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = get_logger(__name__)


class ModelInversionAttack:
    """Black-box model inversion attack to infer sensitive attributes."""

    def __init__(self, target_model: nn.Module, device: torch.device):
        """Initialize MIA.
        
        Args:
            target_model: Target ML model to attack
            device: PyTorch device
        """
        self.target_model = target_model
        self.device = device
        self.target_model.eval()

        # Attacker's surrogate model (neural network)
        self.attacker_model: Optional[nn.Module] = None
        self.attacker_trainer: Optional[AttackerTrainer] = None

        logger.info("ModelInversionAttack initialized")

    def run_attack(
        self,
        X_public: np.ndarray,
        y_sensitive_true: np.ndarray,
        sensitive_attribute_name: str,
        output_mode: str = "probability",
        num_queries: int = 1000,
        batch_size: int = 32,
    ) -> Dict:
        """Execute model inversion attack.
        
        Args:
            X_public: Public/non-sensitive features
            y_sensitive_true: True sensitive attribute values
            sensitive_attribute_name: Name of sensitive attribute being attacked
            output_mode: "probability" or "class" - what the target model exposes
            num_queries: Number of queries to use for attack
            batch_size: Batch size
            
        Returns:
            Attack results dictionary
        """
        logger.info(f"Starting MIA on {sensitive_attribute_name}")
        logger.info(f"  Output mode: {output_mode}")
        logger.info(f"  Num queries: {num_queries}")

        # Sample queries if needed
        if num_queries < len(X_public):
            indices = np.random.choice(len(X_public), size=num_queries, replace=False)
            X_attack = X_public[indices]
            y_attack = y_sensitive_true[indices]
        else:
            X_attack = X_public
            y_attack = y_sensitive_true

        logger.info(f"Using {len(X_attack)} samples for attack")

        # Get target model predictions
        target_predictions = self._query_target_model(X_attack, output_mode, batch_size)

        # Prepare attack dataset
        if output_mode == "probability":
            # Predictions are probabilities
            X_attack_input = np.column_stack([X_attack, target_predictions])
        else:
            # Predictions are class labels (0 or 1)
            X_attack_input = np.column_stack([X_attack, target_predictions.reshape(-1, 1)])

        # Determine if sensitive attribute is categorical or numerical
        is_categorical = isinstance(y_attack[0], (str, np.str_))

        if is_categorical:
            y_attack_encoded, categories = self._encode_categorical(y_attack)
            num_classes = len(categories)
        else:
            y_attack_encoded = y_attack.astype(float)
            categories = None
            num_classes = 1

        # Train attacker model
        logger.info("Training attacker model...")
        attacker_config = AttackerConfig(
            input_dim=X_attack_input.shape[1],
            output_dim=num_classes if is_categorical else 1,
            hidden_dim=128,
            num_layers=2,
            epochs=50,
            batch_size=batch_size,
        )

        self.attacker_model = AttackerMLP(attacker_config)
        self.attacker_trainer = AttackerTrainer(self.attacker_model, attacker_config)
        self.attacker_trainer.train(X_attack_input, y_attack_encoded)

        # Predict sensitive attribute using attacker model
        y_attack_inferred = self._infer_sensitive_attribute(
            X_attack_input,
            is_categorical,
            categories,
        )

        # Evaluate attack
        results = self._evaluate_attack(
            y_attack,
            y_attack_inferred,
            sensitive_attribute_name,
            is_categorical,
        )

        # Add metadata
        results["output_mode"] = output_mode
        results["num_queries"] = len(X_attack)
        results["sensitive_attribute"] = sensitive_attribute_name
        results["is_categorical"] = is_categorical

        logger.info(f"MIA attack complete. Attack accuracy: {results['attack_accuracy']:.4f}")

        return results

    def _query_target_model(
        self,
        X: np.ndarray,
        output_mode: str,
        batch_size: int,
    ) -> np.ndarray:
        """Query target model to get predictions.
        
        Args:
            X: Input features
            output_mode: "probability" or "class"
            batch_size: Batch size
            
        Returns:
            Model predictions
        """
        predictions = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(self.device)

                outputs = self.target_model(X_batch).squeeze()

                if output_mode == "probability":
                    predictions.append(outputs.cpu().numpy())
                else:  # class
                    class_pred = (outputs > 0.5).float().cpu().numpy()
                    predictions.append(class_pred)

        return np.concatenate(predictions, axis=0)

    def _encode_categorical(self, y: np.ndarray) -> Tuple[np.ndarray, List]:
        """Encode categorical sensitive attribute.
        
        Args:
            y: Categorical values
            
        Returns:
            Tuple of (encoded array, categories)
        """
        categories = list(np.unique(y))
        y_encoded = np.array([categories.index(val) for val in y])
        return y_encoded, categories

    def _infer_sensitive_attribute(
        self,
        X_attack: np.ndarray,
        is_categorical: bool,
        categories: Optional[List] = None,
    ) -> np.ndarray:
        """Use attacker model to infer sensitive attribute.
        
        Args:
            X_attack: Attack input (includes target predictions)
            is_categorical: Whether attribute is categorical
            categories: List of categories if categorical
            
        Returns:
            Inferred sensitive attribute values
        """
        self.attacker_model.eval()
        predictions = []

        X_tensor = torch.tensor(X_attack, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.attacker_model(X_tensor)

        if is_categorical:
            # Take argmax for categorical
            predicted_classes = torch.argmax(outputs, dim=1).cpu().numpy()
            if categories:
                inferred = np.array([categories[c] for c in predicted_classes])
            else:
                inferred = predicted_classes
        else:
            # Take raw output for numerical
            inferred = outputs.squeeze().cpu().numpy()

        return inferred

    def _evaluate_attack(
        self,
        y_true: np.ndarray,
        y_inferred: np.ndarray,
        sensitive_attribute_name: str,
        is_categorical: bool,
    ) -> Dict:
        """Evaluate attack performance.
        
        Args:
            y_true: True sensitive attribute values
            y_inferred: Inferred sensitive attribute values
            sensitive_attribute_name: Name of attribute
            is_categorical: Whether categorical
            
        Returns:
            Evaluation dictionary
        """
        results = {
            "sensitive_attribute": sensitive_attribute_name,
        }

        if is_categorical:
            # Classification metrics
            y_true_encoded, categories = self._encode_categorical(y_true)

            if isinstance(y_inferred[0], (str, np.str_)):
                y_inferred_encoded, _ = self._encode_categorical(y_inferred)
            else:
                y_inferred_encoded = y_inferred.astype(int)

            results["attack_accuracy"] = float(accuracy_score(y_true_encoded, y_inferred_encoded))
            results["attack_precision"] = float(
                precision_score(y_true_encoded, y_inferred_encoded, average="weighted", zero_division=0)
            )
            results["attack_recall"] = float(
                recall_score(y_true_encoded, y_inferred_encoded, average="weighted", zero_division=0)
            )
            results["attack_f1"] = float(
                f1_score(y_true_encoded, y_inferred_encoded, average="weighted", zero_division=0)
            )

            cm = confusion_matrix(y_true_encoded, y_inferred_encoded)
            results["confusion_matrix"] = cm.tolist()

            # Distribution matching
            results["true_distribution"] = {
                str(c): int((y_true_encoded == i).sum())
                for i, c in enumerate(categories)
            }
            results["inferred_distribution"] = {
                str(c): int((y_inferred_encoded == i).sum())
                for i, c in enumerate(categories)
            }

        else:
            # Regression metrics
            y_true_float = y_true.astype(float)
            y_inferred_float = y_inferred.astype(float)

            mae = float(np.mean(np.abs(y_true_float - y_inferred_float)))
            rmse = float(np.sqrt(np.mean((y_true_float - y_inferred_float) ** 2)))

            results["mae"] = mae
            results["rmse"] = rmse

            # Approximate attack success as "within reasonable range"
            results["attack_accuracy"] = float(
                1.0 - (mae / (np.max(y_true_float) - np.min(y_true_float) + 1e-6))
            )

        return results


class AttackerConfig:
    """Configuration for attacker model."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        """Initialize config."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size


class AttackerMLP(nn.Module):
    """Neural network for attacker to infer sensitive attribute."""

    def __init__(self, config: AttackerConfig):
        """Initialize model."""
        super().__init__()

        layers = []
        prev_dim = config.input_dim

        for _ in range(config.num_layers):
            layers.append(nn.Linear(prev_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = config.hidden_dim

        layers.append(nn.Linear(prev_dim, config.output_dim))

        if config.output_dim > 1:
            layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class AttackerTrainer:
    """Train attacker model."""

    def __init__(self, model: AttackerMLP, config: AttackerConfig):
        """Initialize trainer."""
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        if config.output_dim > 1:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train attacker model."""
        for epoch in range(self.config.epochs):
            self.model.train()
            indices = np.arange(len(X))
            np.random.shuffle(indices)

            for i in range(0, len(X), self.config.batch_size):
                batch_idx = indices[i : i + self.config.batch_size]
                X_batch = torch.tensor(X[batch_idx], dtype=torch.float32).to(self.device)
                y_batch = torch.tensor(y[batch_idx], dtype=torch.float32).to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(X_batch)

                if self.config.output_dim == 1:
                    outputs = outputs.squeeze()

                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
