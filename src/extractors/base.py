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
    Handles network sessions, rate-limit header parsing, pause & resume state,
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
            self.rate_limit_delay_seconds = 60.0 / float(limit_val)
            logger.info(
                f"Initial API header rate limit detected: {limit_val} requests/window. "
                f"Setting request pacing delay to {self.rate_limit_delay_seconds:.2f} seconds."
            )
        else:
            logger.info("No rate limit header found on initial request; proceeding with default pacing.")

    def enforce_pacing(self) -> None:
        """Enforces request pacing based on rate limit inspection."""
        if self.rate_limit_delay_seconds > 0:
            time.sleep(self.rate_limit_delay_seconds)

    def get_raw_file_path(self, venue: str, year: int) -> Path:
        """Returns standard raw file path data/raw/<venue>/<venue>_<year>.json."""
        venue_dir = self.output_dir / venue.upper()
        venue_dir.mkdir(parents=True, exist_ok=True)
        return venue_dir / f"{venue.upper()}_{year}.json"

    def is_cached(self, venue: str, year: int) -> bool:
        """Checks if raw data file already exists and is marked completed."""
        state = self.get_resume_state(venue, year)
        return state["completed"] and len(state["papers"]) > 0

    def load_cached(self, venue: str, year: int) -> Dict[str, Any]:
        """Loads cached raw data JSON file."""
        path = self.get_raw_file_path(venue, year)
        logger.info(f"Opening file for reading cached raw data: {path.resolve()}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cached raw data file at {path}", exc_info=True)
            raise e

    def get_resume_state(self, venue: str, year: int) -> Dict[str, Any]:
        """Loads existing raw artifact pause/resume state if present."""
        path = self.get_raw_file_path(venue, year)
        if not path.exists() or path.stat().st_size == 0:
            return {"papers": [], "next_cursor": None, "next_page": 1, "completed": False}
        logger.info(f"Opening file for reading resume state: {path.resolve()}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "papers": data.get("papers", []),
                    "next_cursor": data.get("next_cursor"),
                    "next_page": data.get("next_page") or 1,
                    "completed": data.get("completed", False),
                }
        except Exception:
            return {"papers": [], "next_cursor": None, "next_page": 1, "completed": False}

    def save_raw_artifact(
        self,
        venue: str,
        year: int,
        papers: List[Dict[str, Any]],
        next_cursor: Optional[str] = None,
        next_page: Optional[int] = None,
        completed: bool = False,
    ) -> Dict[str, Any]:
        """
        Dumps raw paper data, pagination resume state, and affiliation frequencies into JSON artifact.
        """
        raw_path = self.get_raw_file_path(venue, year)

        all_affiliations: List[str] = []
        for paper in papers:
            all_affiliations.extend(paper.get("raw_affiliations", []))

        freq_counter = Counter(all_affiliations)

        artifact = {
            "venue": venue.upper(),
            "year": year,
            "completed": completed,
            "next_cursor": next_cursor,
            "next_page": next_page,
            "total_papers": len(papers),
            "papers": papers,
            "affiliation_frequencies": dict(freq_counter),
        }

        try:
            logger.info(f"Opening raw data file for writing: {raw_path.resolve()}")
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(artifact, f, indent=2, ensure_ascii=False)
            status_str = "Completed" if completed else "In-progress"
            logger.info(f"Saved raw extraction artifact ({status_str}) to {raw_path} ({len(papers)} papers total)")
            return artifact
        except Exception as e:
            logger.error(f"Failed to write raw extraction artifact to {raw_path}", exc_info=True)
            raise e

    def append_raw_batch(
        self,
        venue: str,
        year: int,
        new_papers: List[Dict[str, Any]],
        next_cursor: Optional[str] = None,
        next_page: Optional[int] = None,
        completed: bool = False,
    ) -> Dict[str, Any]:
        """
        Incrementally appends a new batch of paper records into data/raw/<venue>/<venue>_<year>.json,
        deduplicating by paper_id and flushing updated resume state immediately to disk.
        """
        state = self.get_resume_state(venue, year)
        existing_papers = state.get("papers", [])

        # Deduplicate papers by paper_id
        seen_ids = set()
        merged_papers: List[Dict[str, Any]] = []

        for p in existing_papers + new_papers:
            pid = p.get("paper_id")
            if pid and pid in seen_ids:
                continue
            if pid:
                seen_ids.add(pid)
            merged_papers.append(p)

        return self.save_raw_artifact(
            venue=venue,
            year=year,
            papers=merged_papers,
            next_cursor=next_cursor,
            next_page=next_page,
            completed=completed,
        )

    @abstractmethod
    def extract(self, venue: str, year: int, force: bool = False) -> Dict[str, Any]:
        """
        Extracts raw paper metadata for target venue and year.
        Must be implemented by concrete extractor subclasses.
        """
        pass
