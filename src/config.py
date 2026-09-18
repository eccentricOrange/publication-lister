import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# API Keys & Credentials
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
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

