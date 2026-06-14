"""Configuration module — loads settings from environment variables."""

import os
import logging
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file automatically
load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    deepseek_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    rlm_verbose: bool = field(default_factory=lambda: os.getenv("RLM_VERBOSE", "true").lower() == "true")
    rlm_persistent: bool = True
    rlm_compaction: bool = True
    rlm_compaction_threshold: float = 0.85
    rlm_max_iterations: int = 25
    host: str = field(default_factory=lambda: os.getenv("RLM_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("RLM_PORT", "8000")))

    def __post_init__(self):
        if not self.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. "
                "Set it in your environment or .env file. "
                "Get your key at https://platform.deepseek.com/api_keys"
            )

    @property
    def rlm_backend_kwargs(self) -> dict:
        """Build backend_kwargs dict for RLM constructor."""
        return {
            "api_key": self.deepseek_api_key,
            "model_name": self.deepseek_model,
            "base_url": self.deepseek_base_url,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    settings = Settings()
    logger.info("Settings loaded: model=%s, base_url=%s", settings.deepseek_model, settings.deepseek_base_url)
    return settings
