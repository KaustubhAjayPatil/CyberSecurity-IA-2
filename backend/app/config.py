"""Configuration management for Privacy Shield application."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from .env file
    )

    # Backend Settings
    debug: bool = False
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    reload: bool = False

    # Frontend Settings
    frontend_url: str = "http://localhost:3000"

    # Dataset Settings
    dataset_path: str = "./data/Base.csv"
    dataset_sample_size: Optional[int] = None  # None = use all data
    dataset_random_seed: int = 42

    # ML Training Settings
    ml_random_seed: int = 42
    ml_test_size: float = 0.15
    ml_val_size: float = 0.15

    # Baseline Model Settings
    baseline_epochs: int = 50
    baseline_batch_size: int = 32
    baseline_learning_rate: float = 0.001
    baseline_hidden_layers: int = 3
    baseline_hidden_dim: int = 256
    baseline_dropout: float = 0.3

    # DP-SGD Settings
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-6
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 0.5
    dp_batch_size: int = 32
    dp_epochs: int = 50
    dp_learning_rate: float = 0.001

    # LDP Settings
    ldp_epsilon: float = 1.0

    # Compression Settings
    compression_ratio: float = 0.8

    # MIA Settings
    mia_num_queries: int = 1000
    mia_batch_size: int = 32

    # API Settings
    api_rate_limit: int = 100
    api_rate_limit_window: int = 60

    # Security Settings
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Storage Settings
    storage_path: Path = Path("./data/experiments")
    storage_models_path: Path = Path("./data/models")

    def ensure_paths(self) -> None:
        """Create necessary directories if they don't exist."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.storage_models_path.mkdir(parents=True, exist_ok=True)
        Path("./data").mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()
