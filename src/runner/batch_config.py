import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.utils import sanitize_venue_name

logger = logging.getLogger(__name__)


@dataclass
class VenueException:
    venue: str
    years: Optional[List[int]] = None
    source: Optional[str] = None
    openalex_source_id: Optional[str] = None
    search_term: Optional[str] = None
    doi_prefix: Optional[str] = None
    query_term: Optional[str] = None


@dataclass
class BatchConfig:
    venues: List[str]
    years: List[int]
    default_source: str = "openalex"
    exceptions: List[VenueException] = field(default_factory=list)

    @classmethod
    def from_file(cls, yaml_path: Path) -> "BatchConfig":
        """Loads and parses a YAML batch configuration file."""
        if not yaml_path.exists():
            raise FileNotFoundError(f"Batch configuration file not found at {yaml_path.resolve()}")

        logger.info(f"Loading batch configuration from: {yaml_path.resolve()}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        # 1. Parse venues
        venues_raw = raw_data.get("venues", [])
        if isinstance(venues_raw, str):
            venues = [venues_raw.strip()]
        elif isinstance(venues_raw, list):
            venues = [str(v).strip() for v in venues_raw if str(v).strip()]
        else:
            raise ValueError(f"Invalid 'venues' specified in {yaml_path}: {venues_raw}")

        if not venues:
            raise ValueError(f"No venues specified in batch configuration {yaml_path}")

        # 2. Parse years
        years_raw = raw_data.get("years", {})
        years: List[int] = []
        if isinstance(years_raw, dict):
            start = int(years_raw.get("start", 2020))
            end = int(years_raw.get("end", 2025))
            years = list(range(start, end + 1))
        elif isinstance(years_raw, list):
            years = [int(y) for y in years_raw]
        elif isinstance(years_raw, int):
            years = [years_raw]
        else:
            raise ValueError(f"Invalid 'years' specified in {yaml_path}: {years_raw}")

        # 3. Default source
        default_source = str(raw_data.get("source", "openalex")).lower()

        # 4. Parse exceptions
        exceptions_raw = raw_data.get("exceptions", [])
        exceptions: List[VenueException] = []
        if isinstance(exceptions_raw, list):
            for item in exceptions_raw:
                if isinstance(item, dict) and "venue" in item:
                    ex_venue = str(item["venue"]).strip()
                    ex_years_raw = item.get("years")
                    ex_years = None
                    if isinstance(ex_years_raw, list):
                        ex_years = [int(y) for y in ex_years_raw]
                    elif isinstance(ex_years_raw, int):
                        ex_years = [ex_years_raw]

                    exceptions.append(
                        VenueException(
                            venue=ex_venue,
                            years=ex_years,
                            source=item.get("source"),
                            openalex_source_id=item.get("openalex_source_id"),
                            search_term=item.get("search_term"),
                            doi_prefix=item.get("doi_prefix"),
                            query_term=item.get("query_term"),
                        )
                    )

        return cls(
            venues=venues,
            years=years,
            default_source=default_source,
            exceptions=exceptions,
        )

    def get_overrides(self, venue: str, year: int) -> Dict[str, Any]:
        """Returns merged override parameters for a specific venue and year."""
        clean_target = sanitize_venue_name(venue)
        merged: Dict[str, Any] = {"source": self.default_source}

        for ex in self.exceptions:
            clean_ex = sanitize_venue_name(ex.venue)
            if clean_ex == clean_target or ex.venue.upper() == venue.upper():
                if ex.years is None or year in ex.years:
                    if ex.source:
                        merged["source"] = ex.source.lower()
                    if ex.openalex_source_id:
                        merged["openalex_source_id"] = ex.openalex_source_id
                    if ex.search_term:
                        merged["search_term"] = ex.search_term
                    if ex.doi_prefix:
                        merged["doi_prefix"] = ex.doi_prefix
                    if ex.query_term:
                        merged["query_term"] = ex.query_term

        return merged
