"""Central settings, read from environment variables and .env.

Every module gets paths, seeds and keys from here, never from hard-coded values.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # paths
    data_dir: Path = Path("data")
    ngafid_raw_dir: Path = Path("data/raw/ngafid")
    results_dir: Path = Path("results")

    # dataset source (NGAFID maintenance dataset, Zenodo)
    zenodo_api_url: str = "https://zenodo.org/api"
    zenodo_record_id: str = "6624956"

    # splits and task definition (docs/DECISIONS.md, 2026-10-03)
    n_folds: int = 5
    val_fold: int = 3
    test_fold: int = 4
    binary_window_days: int = 2
    min_flight_seconds: int = 600

    # reproducibility
    random_seed: int = 42
    log_level: str = "INFO"

    # LLM
    google_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    # vector store
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "mtc-maintenance-cases"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # tracing
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "maintenance-triage-copilot"

    # cloud
    aws_region: str = "eu-central-1"
    s3_bucket: str | None = None

    @property
    def sample_dir(self) -> Path:
        return self.data_dir / "sample"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
