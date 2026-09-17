import json
import logging
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from src.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """
    Base Extractor interface for publication venue data sources.
    Handles network sessions, rate-limit header parsing on startup,
    caching raw files under data/raw/<venue>/<venue>_<year>.json,
    and frequency aggregation.
    """

    def __init__(self, output_dir: Path = RAW_DATA_DIR):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.rate_limit_delay_seconds: float = 0.0
        self.rate_limit_detected: Optional[int] = None

    def inspect_rate_limit_headers(self, response: requests.Response) -> None:
        """
        Inspects response headers on initial query to set pacing limit throughout execution run.
        """
        headers = response.headers
        # Check standard rate limit header names
        limit_val = None
        for key in ["x-ratelimit-limit", "ratelimit-limit", "x-rate-limit-limit", "X-RateLimit-Limit"]:
            if key in headers:
                try:
                    limit_val = int(headers[key])
                    break
                except ValueError:
                    pass

        if limit_val and limit_val > 0:
            self.rate_limit_detected = limit_val
            # Calculate delay in seconds assuming limit_val requests per minute or window
            # e.g., if 60 requests per minute, delay = 1.0 sec
            self.rate_limit_delay_seconds = 60.0 / float(limit_val)
            logger.info(
                f"Initial API header rate limit detected: {limit_val} requests/window. "
                f"Setting request pacing delay to {self.rate_limit_delay_seconds:.2f} seconds."
            )
        else:
            logger.info("No rate limit header found on initial request; proceeding with default pacing.")

    def enforce_pacing(self) -> None:
        """Enforces request pacing based on initial API rate limit inspection."""
        if self.rate_limit_delay_seconds > 0:
            time.sleep(self.rate_limit_delay_seconds)

    def get_raw_file_path(self, venue: str, year: int) -> Path:
        """Returns standard raw file path data/raw/<venue>/<venue>_<year>.json."""
        venue_dir = self.output_dir / venue.upper()
        venue_dir.mkdir(parents=True, exist_ok=True)
        return venue_dir / f"{venue.upper()}_{year}.json"

    def is_cached(self, venue: str, year: int) -> bool:
        """Checks if raw data file already exists for target venue and year."""
        path = self.get_raw_file_path(venue, year)
        return path.exists() and path.stat().st_size > 0

    def load_cached(self, venue: str, year: int) -> Dict[str, Any]:
        """Loads cached raw data JSON file."""
        path = self.get_raw_file_path(venue, year)
        logger.info(f"Loading cached raw data from {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cached raw data file at {path}", exc_info=True)
            raise e

    def save_raw_artifact(self, venue: str, year: int, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Dumps raw, un-deduplicated paper data and affiliation frequencies into standardized raw JSON artifact.
        """
        raw_path = self.get_raw_file_path(venue, year)

        # Aggregate raw affiliation frequencies across all papers
        all_affiliations: List[str] = []
        for paper in papers:
            all_affiliations.extend(paper.get("raw_affiliations", []))

        freq_counter = Counter(all_affiliations)

        artifact = {
            "venue": venue.upper(),
            "year": year,
            "total_papers": len(papers),
            "papers": papers,
            "affiliation_frequencies": dict(freq_counter),
        }

        try:
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(artifact, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved raw extraction artifact to {raw_path} ({len(papers)} papers)")
            return artifact
        except Exception as e:
            logger.error(f"Failed to write raw extraction artifact to {raw_path}", exc_info=True)
            raise e

    @abstractmethod
    def extract(self, venue: str, year: int, force: bool = False) -> Dict[str, Any]:
        """
        Extracts raw paper metadata for target venue and year.
        Must be implemented by concrete extractor subclasses.
        """
        pass
