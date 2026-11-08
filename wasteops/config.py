from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    app_name: str = "WasteOps"
    model_backend: str = "auto"
    model_id: str = "openai/clip-vit-base-patch32"
    model_path: str = "models/wasteops_mobilenetv3.pt"
    device: str = "auto"
    max_upload_mb: int = 10
    min_score: float = 0.24
    min_margin: float = 0.025
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    @property
    def checkpoint_path(self) -> Path:
        p = Path(self.model_path)
        return p if p.is_absolute() else ROOT / p

@lru_cache
def get_settings() -> Settings:
    return Settings()
