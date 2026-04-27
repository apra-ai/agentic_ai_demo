from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_deployment: str
    azure_openai_api_version: str
    docs_dir: Path
    max_iterations: int = 5


REQUIRED_ENV_VARS = {
    "AZURE_OPENAI_API_KEY": "azure_openai_api_key",
    "AZURE_OPENAI_ENDPOINT": "azure_openai_endpoint",
    "AZURE_OPENAI_DEPLOYMENT": "azure_openai_deployment",
    "AZURE_OPENAI_API_VERSION": "azure_openai_api_version",
}


def load_settings() -> Settings:
    """Load and validate the demo configuration from environment variables."""
    load_dotenv()

    values: dict[str, str] = {}
    missing: list[str] = []

    for env_name, field_name in REQUIRED_ENV_VARS.items():
        value = os.getenv(env_name)
        if not value:
            missing.append(env_name)
            continue
        values[field_name] = value

    if missing:
        formatted = ", ".join(missing)
        raise ValueError(
            f"Missing required environment variables: {formatted}. "
            "Create a .env file based on .env.example."
        )

    project_root = Path(__file__).resolve().parent
    return Settings(
        docs_dir=project_root / "docs",
        **values,
    )
