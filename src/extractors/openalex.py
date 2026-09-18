import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from src.config import (
    OPENALEX_API_KEY,
    OPENALEX_MAILTO,
    OPENALEX_SOURCES_CACHE_PATH,
)
from src.extractors.base import BaseExtractor
from src.normalizer.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_SOURCES_URL = "https://api.openalex.org/sources"


DEFAULT_DOI_PREFIXES: Dict[str, str] = {
    "ICRA": "10.1109/icra",
    "IROS": "10.1109/iros",
    "CVPR": "10.1109/cvpr",
    "RA-L": "10.1109/lra",
    "R-AL": "10.1109/lra",
    "RAL": "10.1109/lra",
    "TRO": "10.1109/tro",
    "T-RO": "10.1109/tro",
}
def derive_doi_prefix(venue_str: str, explicit_prefix: Optional[str] = None) -> Optional[str]:
    """
    Derives DOI prefix dynamically from explicit parameter or venue name string in YAML.
    - If explicit_prefix is provided, returns explicit_prefix.
    - Extracts short tag/acronym from venue_str (e.g. 'ICRA (International...)' -> 'ICRA').
    - Cleans non-alphanumeric characters (e.g. 'T-RO' -> 'tro', 'R-AL' -> 'ral', 'ICRA' -> 'icra').
    - Constructs '10.1109/{clean_tag}' as candidate IEEE DOI prefix.
    """
    if explicit_prefix and str(explicit_prefix).strip():
        return str(explicit_prefix).strip()

    if not venue_str:
        return None

    # Extract short tag (part before parentheses if present)
    raw_tag = venue_str.split("(")[0].strip()
    clean_tag = "".join(c.lower() for c in raw_tag if c.isalnum())
    if not clean_tag:
        return None

    return f"10.1109/{clean_tag}"


class OpenAlexExtractor(BaseExtractor):
    """
    OpenAlex REST API extractor implementation.
    Harvests publication metadata and institutional affiliations using exact Source IDs or DOI prefixes.
    - Resolves OpenAlex Source ID via Sources API + Gemini LLM if un-cached.
    - Caches source IDs in data/openalex_sources_cache.json.
    - Queries Works API using primary_location.source.id or doi_starts_with (never unconstrained raw text search).
    - Queries Works API using primary_location.source.id or dynamically derived doi_starts_with.
    - Supports deterministic pause-and-resume via cursor tokens.
    """

    def __init__(
        self,
        api_key: str = OPENALEX_API_KEY,
        mailto: str = OPENALEX_MAILTO,
        sources_cache_path: Path = OPENALEX_SOURCES_CACHE_PATH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.mailto = mailto
        self.sources_cache_path = sources_cache_path
        self.gemini_client = GeminiClient()
        self.rate_limit_delay_seconds = 0.15

    def _load_sources_cache(self) -> Dict[str, str]:
        """Loads cached venue -> OpenAlex Source ID map."""
        if not self.sources_cache_path.exists() or self.sources_cache_path.stat().st_size == 0:
            return {}
        try:
            logger.info(f"Opening file for reading OpenAlex sources cache: {self.sources_cache_path.resolve()}")
            with open(self.sources_cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_sources_cache(self, cache: Dict[str, str]) -> None:
        """Saves venue -> OpenAlex Source ID map to disk."""
        try:
            self.sources_cache_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Opening file for writing OpenAlex sources cache: {self.sources_cache_path.resolve()}")
            with open(self.sources_cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed writing OpenAlex sources cache file: {e}")

    def resolve_source_id(
        self,
        venue: str,
        search_term: Optional[str] = None,
        openalex_source_id: Optional[str] = None,
    ) -> str:
        """
        Resolves OpenAlex Source ID for a given venue abbreviation or full name.
        Uses explicit openalex_source_id if provided; otherwise checks local cache or queries OpenAlex Sources API.
        All candidate results from OpenAlex Sources API are evaluated by Gemini LLM.
        """
        venue_upper = venue.upper()

        if openalex_source_id:
            clean_id = openalex_source_id.split("/")[-1]
            logger.info(f"Using explicitly configured OpenAlex Source ID for venue '{venue_upper}': {clean_id}")
            cache = self._load_sources_cache()
            cache[venue_upper] = clean_id
            self._save_sources_cache(cache)
            return clean_id

        cache = self._load_sources_cache()
        if venue_upper in cache:
            cached_id = cache[venue_upper]
            logger.info(f"Using cached OpenAlex Source ID for venue '{venue_upper}': {cached_id}")
            return cached_id

        venue_search_term = search_term or venue_upper
        logger.info(f"Resolving OpenAlex Source ID for venue '{venue_upper}' ('{venue_search_term}') via Sources API...")

        headers: Dict[str, str] = {}
        if self.api_key:
            headers["api-key"] = self.api_key

        params: Dict[str, Any] = {"search": venue_search_term}
        if self.api_key:
            params["api_key"] = self.api_key
        elif self.mailto:
            params["mailto"] = self.mailto

        prepared_url = requests.Request("GET", OPENALEX_SOURCES_URL, headers=headers, params=params).prepare().url
        logger.info(f"Querying OpenAlex Sources URL: {prepared_url}")

        self.enforce_pacing()
        response = self.session.get(OPENALEX_SOURCES_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        raw_results = data.get("results", [])

        if not raw_results:
            err_msg = f"No OpenAlex sources found for venue search term '{venue_search_term}'"
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Filter out candidates with 0 works if candidates with works > 0 exist
        results = [c for c in raw_results if (c.get("works_count") or 0) > 0]
        if not results:
            results = raw_results

        logger.info(f"Found {len(results)} candidate sources for '{venue_upper}'. Invoking Gemini LLM to select best match...")
        gemini_selected = self.gemini_client.resolve_openalex_source(venue_search_term, results)
        if gemini_selected:
            source_id = gemini_selected
        elif len(results) == 1:
            raw_id = results[0].get("id", "")
            source_id = raw_id.split("/")[-1]
            logger.info(f"Fallback: using single candidate source ID '{source_id}' for '{venue_upper}'")
        else:
            best_cand = max(results, key=lambda x: x.get("works_count", 0))
            source_id = best_cand.get("id", "").split("/")[-1]
            logger.info(f"Fallback selected source ID '{source_id}' ({best_cand.get('display_name')}) with max works_count={best_cand.get('works_count')}")

        cache[venue_upper] = source_id
        self._save_sources_cache(cache)
        logger.info(f"Successfully cached OpenAlex Source ID for venue '{venue_upper}': {source_id}")
        return source_id

    def _determine_filter_params(
        self,
        venue_upper: str,
        year: int,
        source_id: str,
        headers: Dict[str, str],
        search_term: Optional[str] = None,
        doi_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Determines optimal filter parameters for OpenAlex Works API.
        Tries primary_location.source.id first; if 0 results returned, falls back to doi_starts_with.
        Tries primary_location.source.id first; if 0 results returned, falls back to dynamically derived doi_starts_with.
        STRICT REQUIREMENT: Never falls back to unconstrained raw text search ('search: {venue}').
        """
        effective_doi_prefix = doi_prefix or DEFAULT_DOI_PREFIXES.get(venue_upper)
        if not effective_doi_prefix:
            # Try cleaning venue name (e.g., 'T-RO' -> 'TRO', 'R-AL' -> 'RAL')
            clean_key = venue_upper.replace(" ", "").replace("-", "").replace("_", "")
            effective_doi_prefix = DEFAULT_DOI_PREFIXES.get(clean_key)
        effective_doi_prefix = derive_doi_prefix(venue_upper, explicit_prefix=doi_prefix)

        test_filter = f"publication_year:{year},primary_location.source.id:{source_id}"
        test_params: Dict[str, Any] = {
            "filter": test_filter,
            "per-page": 1,
        }
        if self.api_key:
            test_params["api_key"] = self.api_key
        elif self.mailto:
            test_params["mailto"] = self.mailto

        self.enforce_pacing()
        try:
            res = self.session.get(OPENALEX_WORKS_URL, headers=headers, params=test_params, timeout=30)
            if res.status_code == 200:
                cnt = res.json().get("meta", {}).get("count", 0)
                if cnt > 0:
                    logger.info(f"OpenAlex filter strategy for {venue_upper} {year}: primary_location.source.id:{source_id} (count={cnt})")
                    return {"filter": test_filter}
        except Exception as e:
            logger.warning(f"Failed testing primary_location.source.id filter for {venue_upper} {year}: {e}")

        # Fallback 1: doi_starts_with
        # Fallback 1: doi_starts_with (derived dynamically from venue string or explicit config)
        if effective_doi_prefix:
            doi_filter = f"publication_year:{year},doi_starts_with:{effective_doi_prefix}"
            test_params["filter"] = doi_filter
            self.enforce_pacing()
            try:
                res = self.session.get(OPENALEX_WORKS_URL, headers=headers, params=test_params, timeout=30)
                if res.status_code == 200:
                    cnt = res.json().get("meta", {}).get("count", 0)
                    if cnt > 0:
                        logger.info(f"OpenAlex filter strategy for {venue_upper} {year}: doi_starts_with:{effective_doi_prefix} (count={cnt})")
                        return {"filter": doi_filter}
            except Exception as e:
                logger.warning(f"Failed testing doi_starts_with filter for {venue_upper} {year}: {e}")

        # Strict: If count is 0 for both source_id and doi_starts_with, DO NOT fall back to unconstrained raw text search.
        if effective_doi_prefix:
            fallback_filter = f"publication_year:{year},doi_starts_with:{effective_doi_prefix}"
        else:
            fallback_filter = f"publication_year:{year},primary_location.source.id:{source_id}"

        logger.warning(
            f"No OpenAlex works found for {venue_upper} {year} with source ID '{source_id}' or DOI prefix '{effective_doi_prefix}'. "
            f"No OpenAlex works found for {venue_upper} {year} with source ID '{source_id}' or derived DOI prefix '{effective_doi_prefix}'. "
            f"Using filter '{fallback_filter}' (count=0) to prevent unconstrained raw search corruption."
        )
        return {"filter": fallback_filter}

    def extract(
        self,
        venue: str,
        year: int,
        force: bool = False,
        search_term: Optional[str] = None,
        openalex_source_id: Optional[str] = None,
        doi_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.api_key and not self.mailto:
            err_msg = "Neither OPENALEX_API_KEY nor OPENALEX_MAILTO is set in environment or configuration. Cannot proceed with OpenAlex extraction."
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        venue_upper = venue.upper()
        raw_path = self.get_raw_file_path(venue_upper, year)

        if not force and self.is_cached(venue_upper, year):
            logger.info(f"Raw data for {venue_upper} {year} is fully cached and completed.")
            return self.load_cached(venue_upper, year)

        source_id = self.resolve_source_id(venue_upper, search_term=search_term, openalex_source_id=openalex_source_id)

        cursor = "*"
        existing_papers_count = 0
        if not force and raw_path.exists() and raw_path.stat().st_size > 0:
            try:
                cached_data = self.load_cached(venue_upper, year)
                if not cached_data.get("completed", False):
                    cursor = cached_data.get("next_cursor", "*")
                    existing_papers_count = cached_data.get("total_papers", 0)
                    logger.info(f"Resuming partial OpenAlex extraction for {venue_upper} {year} from cursor '{cursor}' ({existing_papers_count} papers already saved)")
            except Exception as e:
                logger.warning(f"Could not load existing partial state from {raw_path}: {e}")

        headers: Dict[str, str] = {}
        if self.api_key:
            headers["api-key"] = self.api_key

        base_filter_params = self._determine_filter_params(
            venue_upper, year, source_id, headers, search_term=search_term, doi_prefix=doi_prefix
        )

        select_fields = "id,doi,title,authorships,primary_location"

        logger.info(f"Starting OpenAlex harvesting for venue '{venue_upper}' ({year}) from cursor '{cursor}'...")

        page_count = 0
        total_extracted_this_run = 0

        while cursor:
            params: Dict[str, Any] = dict(base_filter_params)
            params.update({
                "per-page": 200,
                "cursor": cursor,
                "select": select_fields,
            })
            if self.api_key:
                params["api_key"] = self.api_key
            elif self.mailto:
                params["mailto"] = self.mailto

            prepared_url = requests.Request("GET", OPENALEX_WORKS_URL, headers=headers, params=params).prepare().url
            logger.info(f"Querying OpenAlex Works URL (Page {page_count + 1}): {prepared_url}")

            response_data = None
            for attempt in range(5):
                self.enforce_pacing()
                try:
                    response = self.session.get(OPENALEX_WORKS_URL, headers=headers, params=params, timeout=45)
                    if response.status_code in (429, 503):
                        sleep_time = (2 ** attempt) * 2.0 + random.uniform(0.5, 1.5)
                        logger.warning(f"OpenAlex API rate limited/busy ({response.status_code}). Sleeping {sleep_time:.2f}s (Attempt {attempt+1}/5)")
                        time.sleep(sleep_time)
                        continue
                    response.raise_for_status()
                    response_data = response.json()
                    break
                except Exception as e:
                    if attempt < 4:
                        sleep_time = (2 ** attempt) * 2.0 + random.uniform(0.5, 1.5)
                        logger.warning(f"Request error querying OpenAlex Works ({e}). Retrying in {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Exhausted retries harvesting OpenAlex Works for {venue_upper} {year} at cursor '{cursor}': {e}", exc_info=True)
                        raise e

            if not response_data:
                raise RuntimeError(f"Failed receiving data from OpenAlex API for {venue_upper} {year}")

            meta = response_data.get("meta", {})
            next_cursor = meta.get("next_cursor")
            results = response_data.get("results", [])

            paper_batch: List[Dict[str, Any]] = []
            for work in results:
                raw_work_id = work.get("id", "")
                paper_id = raw_work_id.split("/")[-1] if raw_work_id else ""
                title = work.get("title") or ""
                authorships = work.get("authorships", [])

                raw_affiliations: List[str] = []
                for auth in authorships:
                    raw_affs = auth.get("raw_affiliation_strings", [])
                    if raw_affs:
                        for aff_str in raw_affs:
                            if aff_str and str(aff_str).strip():
                                raw_affiliations.append(str(aff_str).strip())
                    else:
                        insts = auth.get("institutions", [])
                        for inst in insts:
                            display_name = inst.get("display_name")
                            if display_name and str(display_name).strip():
                                raw_affiliations.append(str(display_name).strip())

                paper_batch.append({
                    "paper_id": paper_id,
                    "title": title,
                    "raw_affiliations": raw_affiliations,
                })

            page_count += 1
            total_extracted_this_run += len(paper_batch)

            is_completed = (next_cursor is None or len(results) == 0)
            checkpoint = self.append_raw_batch(
                venue_upper,
                year,
                paper_batch,
                completed=is_completed,
                next_cursor=next_cursor,
            )

            logger.info(f"OpenAlex Harvested Page {page_count}: +{len(paper_batch)} papers (Total: {checkpoint['total_papers']}). Next Cursor: '{next_cursor}'")

            if is_completed:
                logger.info(f"Successfully completed OpenAlex harvesting for {venue_upper} {year} ({checkpoint['total_papers']} total papers).")
                return checkpoint

            cursor = next_cursor

        return self.load_cached(venue_upper, year)
