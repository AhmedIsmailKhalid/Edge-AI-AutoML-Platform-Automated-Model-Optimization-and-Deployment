"""
Application configuration module.
Loads and validates environment variables.
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Database - Read from environment variable
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./experiments.db")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    # CORS - Allow Vercel frontend
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")

    # File Storage
    dataset_dir: str = "dataset"
    models_dir: str = "models"

    # Application
    debug: bool = False
    log_level: str = "INFO"

    # Security
    secret_key: str = os.getenv("SECRET_KEY", "change-me-in-production")

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def dataset_path(self) -> Path:
        """Get dataset directory path."""
        return Path(self.dataset_dir)

    @property
    def models_path(self) -> Path:
        """Get models directory path."""
        return Path(self.models_dir)

    @property
    def preset_dataset_path(self) -> Path:
        """Get preset dataset directory path."""
        return self.dataset_path / "preset"

    @property
    def custom_dataset_path(self) -> Path:
        """Get custom dataset directory path."""
        return self.dataset_path / "custom"

    @property
    def pretrained_models_path(self) -> Path:
        """Get pretrained models directory path."""
        return self.models_path / "pretrained"

    @property
    def custom_models_path(self) -> Path:
        """Get custom models directory path."""
        return self.models_path / "custom"

    @property
    def optimized_models_path(self) -> Path:
        """Get optimized models directory path."""
        return self.models_path / "optimized"

    @property
    def custom_datasets_path(self) -> Path:
        """Path to custom datasets directory."""
        path = Path("dataset/custom")
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global settings instance
settings = Settings()
