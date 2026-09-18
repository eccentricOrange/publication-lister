import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.utils import sanitize_venue_name

logger = logging.getLogger(__name__)


@dataclass
class VenueConfig:
    search_term: Any  # str or List[str]
    short_name: str
    openalex_source_id: Optional[Any] = None
    doi_prefix: Optional[Any] = None
    query_term: Optional[Any] = None
    source: Optional[str] = None


@dataclass
class VenueException:
    venue: str
    years: Optional[List[int]] = None
    source: Optional[str] = None
    openalex_source_id: Optional[Any] = None
    search_term: Optional[Any] = None  # str or List[str]
    doi_prefix: Optional[Any] = None
    query_term: Optional[Any] = None


@dataclass
class BatchConfig:
    venue_configs: List[VenueConfig]
    years: List[int]
    default_source: str = "openalex"
    exceptions: List[VenueException] = field(default_factory=list)

    @property
    def venues(self) -> List[str]:
        """Returns list of short_name strings for backward compatibility."""
        return [v.short_name for v in self.venue_configs]

    @classmethod
    def from_file(cls, yaml_path: Path) -> "BatchConfig":
        """Loads and parses a YAML batch configuration file."""
        if not yaml_path.exists():
            raise FileNotFoundError(f"Batch configuration file not found at {yaml_path.resolve()}")

        logger.info(f"Loading batch configuration from: {yaml_path.resolve()}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        # 1. Parse venues
        raw_venues = raw_data.get("venues", [])
        venue_configs: List[VenueConfig] = []

        def parse_search_term(raw_val: Any) -> Any:
            if isinstance(raw_val, list):
                clean_list = [str(x).strip() for x in raw_val if str(x).strip()]
                return clean_list if clean_list else ""
            if isinstance(raw_val, str):
                return raw_val.strip()
            return ""

        if isinstance(raw_venues, list):
            for item in raw_venues:
                if isinstance(item, str) and item.strip():
                    st = item.strip()
                    sn = sanitize_venue_name(st)
                    venue_configs.append(VenueConfig(search_term=st, short_name=sn))
                elif isinstance(item, dict):
                    st_raw = item.get("search_term") or item.get("search_terms") or item.get("search") or item.get("name") or item.get("venue")
                    st = parse_search_term(st_raw)
                    if not st:
                        continue
                    sn_raw = (item.get("short_name") or item.get("short_code") or item.get("code") or item.get("short") or "").strip()
                    first_st = st[0] if isinstance(st, list) else st
                    sn = sanitize_venue_name(sn_raw) if sn_raw else sanitize_venue_name(first_st)
                    venue_configs.append(
                        VenueConfig(
                            search_term=st,
                            short_name=sn,
                            openalex_source_id=item.get("openalex_source_id"),
                            doi_prefix=item.get("doi_prefix"),
                            query_term=item.get("query_term"),
                            source=item.get("source"),
                        )
                    )
        elif isinstance(raw_venues, str) and raw_venues.strip():
            st = raw_venues.strip()
            sn = sanitize_venue_name(st)
            venue_configs.append(VenueConfig(search_term=st, short_name=sn))

        if not venue_configs:
            raise ValueError(f"No valid venues specified in batch configuration {yaml_path}")

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

                    ex_st_raw = item.get("search_term") or item.get("search_terms") or item.get("search")
                    ex_st = parse_search_term(ex_st_raw) if ex_st_raw else None

                    exceptions.append(
                        VenueException(
                            venue=ex_venue,
                            years=ex_years,
                            source=item.get("source"),
                            openalex_source_id=item.get("openalex_source_id"),
                            search_term=ex_st,
                            doi_prefix=item.get("doi_prefix"),
                            query_term=item.get("query_term"),
                        )
                    )

        return cls(
            venue_configs=venue_configs,
            years=years,
            default_source=default_source,
            exceptions=exceptions,
        )

    def get_overrides(self, venue: str, year: int) -> Dict[str, Any]:
        """Returns merged override parameters for a specific venue (by short_name or search_term) and year."""
        clean_target = sanitize_venue_name(venue)
        merged: Dict[str, Any] = {"source": self.default_source}

        # 1. Base configuration from VenueConfig
        for v_cfg in self.venue_configs:
            if v_cfg.short_name == clean_target or sanitize_venue_name(v_cfg.search_term) == clean_target:
                merged["search_term"] = v_cfg.search_term
                if v_cfg.source:
                    merged["source"] = v_cfg.source.lower()
                if v_cfg.openalex_source_id:
                    merged["openalex_source_id"] = v_cfg.openalex_source_id
                if v_cfg.doi_prefix:
                    merged["doi_prefix"] = v_cfg.doi_prefix
                if v_cfg.query_term:
                    merged["query_term"] = v_cfg.query_term
                break

        # 2. Exceptions override
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

    def is_year_active(self, venue: str, year: int) -> bool:
        """Checks if a year is explicitly active for a venue based on YAML exceptions."""
        clean_target = sanitize_venue_name(venue)
        for ex in self.exceptions:
            clean_ex = sanitize_venue_name(ex.venue)
            if clean_ex == clean_target or ex.venue.upper() == venue.upper():
                if ex.years is not None and year not in ex.years:
                    return False
        return True
