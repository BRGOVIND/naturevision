"""Application configuration sourced from the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    """Runtime configuration. Every value is overridable through the environment."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- application ---------------------------------------------------
    app_name: str = "NatureVision"
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="console", description="console | json")
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default=["http://localhost:5173", "http://localhost:3000"])

    # --- database ------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://naturevision:naturevision@localhost:5432/naturevision"
    )
    db_echo: bool = False

    # --- imagery provider ----------------------------------------------
    stac_endpoint: str = "https://earth-search.aws.element84.com/v1"
    stac_collection: str = "sentinel-2-l2a"
    stac_timeout_seconds: float = 60.0
    max_cloud_cover_percent: float = Field(default=40.0, ge=0.0, le=100.0)
    max_search_results: int = Field(default=50, ge=1, le=500)
    stac_max_retries: int = Field(default=4, ge=1, le=10)
    stac_max_concurrent_requests: int = Field(default=4, ge=1, le=32)

    # --- raster processing ----------------------------------------------
    max_region_area_km2: float = Field(default=2500.0, gt=0)
    min_region_area_km2: float = Field(default=0.01, gt=0)
    target_raster_max_dim: int = Field(
        default=768, ge=64, le=4096, description="Longest edge of the analysis grid in pixels."
    )
    raster_cache_dir: Path = Field(default=REPO_ROOT / ".cache" / "raster")
    http_raster_retries: int = 3

    # --- change detection -----------------------------------------------
    change_moderate_threshold: float = Field(default=0.10, gt=0, lt=2)
    change_significant_threshold: float = Field(default=0.20, gt=0, lt=2)

    # --- machine learning ------------------------------------------------
    model_dir: Path = Field(default=BACKEND_ROOT / "artifacts" / "models")
    land_cover_backend: str = Field(
        default="random_forest", description="random_forest | torch_mlp"
    )

    # --- language interpretation ------------------------------------------
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    language_model: str = "llama-3.3-70b-versatile"
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    language_timeout_seconds: float = 90.0
    language_max_tokens: int = 2000
    language_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    enable_vision_interpretation: bool = True

    # --- storage -----------------------------------------------------------
    artifact_dir: Path = Field(default=BACKEND_ROOT / "artifacts" / "analyses")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")

    @property
    def sync_database_url(self) -> str:
        """Synchronous DSN, used by Alembic and management commands."""
        return self.database_url.replace("+asyncpg", "+psycopg").replace("+aiosqlite", "")

    @property
    def language_enabled(self) -> bool:
        return bool(self.groq_api_key)

    def ensure_directories(self) -> None:
        for directory in (self.raster_cache_dir, self.model_dir, self.artifact_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
