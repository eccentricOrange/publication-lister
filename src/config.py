import os
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# API Keys & Credentials
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
IEEE_API_KEY = os.getenv("IEEE_API_KEY", "")
SCOPUS_API_KEY = os.getenv("SCOPUS_API_KEY", "")

# Directory Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
NORMALIZED_DATA_DIR = DATA_DIR / "normalized"
OUTPUT_DATA_DIR = DATA_DIR / "output"
CLEANED_OUTPUT_DATA_DIR = DATA_DIR / "cleaned_output"
CANONICAL_REGISTRY_PATH = DATA_DIR / "canonical_organizations.json"
OPENALEX_SOURCES_CACHE_PATH = DATA_DIR / "openalex_sources_cache.json"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
NORMALIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_gemini_model(
    cli_model: Optional[str] = None,
    config_path: Optional[Path] = None,
    batch_config: Optional[Any] = None,
) -> str:
    """
    Resolves the Gemini model string according to strict 3-step priority order:
    1. CLI flag (--model / -m) if explicitly provided and non-empty.
    2. YAML configuration file model setting (from batch_config or batch.yaml file).
    3. Hardcoded default (DEFAULT_GEMINI_MODEL from config / env).
    """
    if cli_model and str(cli_model).strip():
        return str(cli_model).strip()

    if batch_config and getattr(batch_config, "model", None):
        return str(batch_config.model).strip()

    target_yaml = config_path or (BASE_DIR / "batch.yaml")
    if target_yaml and Path(target_yaml).exists():
        try:
            import yaml
            with open(target_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                m = data.get("model")
                if m and str(m).strip():
                    return str(m).strip()
        except Exception:
            pass

    return DEFAULT_GEMINI_MODEL


