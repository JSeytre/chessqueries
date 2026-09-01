"""Project settings and filesystem layout.

Values load from `.env` (via pydantic-settings) and the process environment.
Everything is anchored to the repo root so paths work regardless of cwd.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (chessqueries/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=True)

    # Root for all downloaded datasets and the checkpoints cache.
    DATA_ROOT: Path = REPO_ROOT / "data"
    CHECKPOINTS_ROOT: Path = REPO_ROOT / "checkpoints"
    OUTPUTS_ROOT: Path = REPO_ROOT / "outputs"

    # Credentials for the prompted-VLM baseline (see `.env.example`). Optional:
    # left unset, each SDK falls back to its own environment variable, so an
    # existing shell export keeps working. `SecretStr` keeps the value out of
    # reprs and tracebacks -- these end up in logs we paste around.
    ANTHROPIC_API_KEY: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None

    # Where a self-hosted OpenAI-compatible server (vLLM etc.) is listening, for
    # `Provider.LOCAL`. No credential: the server ignores it.
    LOCAL_BASE_URL: str = "http://localhost:8000/v1"

    @property
    def chessred_root(self) -> Path:
        return self.DATA_ROOT / "chessred"

    @property
    def chesscog_root(self) -> Path:
        return self.DATA_ROOT / "chesscog"

    @property
    def cvchess_root(self) -> Path:
        return self.DATA_ROOT / "cvchess"


@lru_cache(maxsize=1)
def get_config() -> Settings:
    return Settings()
