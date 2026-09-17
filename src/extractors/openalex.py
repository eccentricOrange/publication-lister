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

VENUE_SEARCH_MAP = {
    "ICRA": "IEEE International Conference on Robotics and Automation",
    "IROS": "IEEE/RSJ International Conference on Intelligent Robots and Systems",
    "RA-L": "IEEE Robotics and Automation Letters",
    "TRO": "IEEE Transactions on Robotics",
    "CVPR": "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
    "NEURIPS": "Neural Information Processing Systems",
    "ECCV": "European Conference on Computer Vision",
    "CORL": "Conference on Robot Learning",
}

VENUE_DOI_PREFIX_MAP = {
    "ICRA": "10.1109/icra",
    "IROS": "10.1109/iros",
    "RA-L": "10.1109/lra",
    "TRO": "10.1109/tro",
    "CVPR": "10.1109/cvpr",
}


class OpenAlexExtractor(BaseExtractor):
    """
    OpenAlex REST API extractor implementation.
    Harvests publication metadata and institutional affiliations using exact Source IDs or DOI prefixes.
    - Resolves OpenAlex Source ID via Sources API + Gemini LLM if un-cached.
    - Caches source IDs in data/openalex_sources_cache.json.
    - Queries Works API using primary_location.source.id, fallback to doi_starts_with or search, with field selection.
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

    def resolve_source_id(self, venue: str) -> str:
        """
        Resolves OpenAlex Source ID for a given venue abbreviation or full name.
        Uses cached value if present; otherwise queries OpenAlex Sources API and uses Gemini LLM.
        """
        venue_upper = venue.upper()
        cache = self._load_sources_cache()

        if venue_upper in cache:
            cached_id = cache[venue_upper]
            logger.info(f"Using cached OpenAlex Source ID for venue '{venue_upper}': {cached_id}")
            return cached_id

        venue_search_term = VENUE_SEARCH_MAP.get(venue_upper, venue)
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

        if len(results) == 1:
            raw_id = results[0].get("id", "")
            source_id = raw_id.split("/")[-1]
            logger.info(f"Single candidate found. Resolved OpenAlex Source ID for '{venue_upper}': {source_id}")
        else:
            logger.info(f"Found {len(results)} candidate sources for '{venue_upper}'. Invoking Gemini LLM to select best match...")
            gemini_selected = self.gemini_client.resolve_openalex_source(venue_search_term, results)
            if gemini_selected:
                source_id = gemini_selected
            else:
                best_cand = max(results, key=lambda x: x.get("works_count", 0))
                source_id = best_cand.get("id", "").split("/")[-1]
                logger.info(f"Fallback selected source ID '{source_id}' ({best_cand.get('display_name')}) with max works_count={best_cand.get('works_count')}")

        cache[venue_upper] = source_id
        self._save_sources_cache(cache)
        logger.info(f"Successfully cached OpenAlex Source ID for venue '{venue_upper}': {source_id}")
        return source_id

    def _determine_filter_params(self, venue_upper: str, year: int, source_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Determines the optimal filter parameters for OpenAlex Works API.
        Tries primary_location.source.id first; if 0 results returned, falls back to doi_starts_with or search.
        """
        # Test primary_location.source.id
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
        except Exception:
            pass

        # Fallback 1: doi_starts_with
        doi_prefix = VENUE_DOI_PREFIX_MAP.get(venue_upper)
        if doi_prefix:
            doi_filter = f"publication_year:{year},doi_starts_with:{doi_prefix}"
            test_params["filter"] = doi_filter
            self.enforce_pacing()
            try:
                res = self.session.get(OPENALEX_WORKS_URL, headers=headers, params=test_params, timeout=30)
                if res.status_code == 200:
                    cnt = res.json().get("meta", {}).get("count", 0)
                    if cnt > 0:
                        logger.info(f"OpenAlex filter strategy for {venue_upper} {year}: doi_starts_with:{doi_prefix} (count={cnt})")
                        return {"filter": doi_filter}
            except Exception:
                pass

        # Fallback 2: text search
        venue_name = VENUE_SEARCH_MAP.get(venue_upper, venue_upper)
        logger.info(f"OpenAlex filter strategy for {venue_upper} {year}: publication_year:{year} with search='{venue_name}'")
        return {"filter": f"publication_year:{year}", "search": venue_name}

    def extract(self, venue: str, year: int, force: bool = False) -> Dict[str, Any]:
        if not self.api_key and not self.mailto:
            err_msg = "Neither OPENALEX_API_KEY nor OPENALEX_MAILTO is set in environment or configuration. Cannot proceed with OpenAlex extraction."
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        venue_upper = venue.upper()
        raw_path = self.get_raw_file_path(venue_upper, year)

        if not force and self.is_cached(venue_upper, year):
            logger.info(f"Raw data for {venue_upper} {year} is fully cached and completed.")
            return self.load_cached(venue_upper, year)

        source_id = self.resolve_source_id(venue_upper)

        cursor = "*"
        existing_papers_count = 0
        if not force and raw_path.exists() and raw_path.stat().st_size > 0:
            try:
                cached_data = self.load_cached(venue_upper, year)
                if not cached_data.get("completed", False):
                    cursor = cached_data.get("next_cursor") or "*"
                    existing_papers_count = cached_data.get("total_papers", 0)
                    logger.info(f"Resuming OpenAlex extraction for {venue_upper} {year} from paper #{existing_papers_count + 1} (next_cursor: {cursor})")
            except Exception:
                cursor = "*"

        logger.info(f"Harvesting OpenAlex works for venue '{venue_upper}' (source_id: {source_id}) year {year}")

        initial_request_done = False
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["api-key"] = self.api_key

        filter_base_params = self._determine_filter_params(venue_upper, year, source_id, headers)

        while cursor:
            params: Dict[str, Any] = {
                "select": "id,doi,title,display_name,publication_year,publication_date,authorships,primary_location,biblio",
                "per-page": 200,
                "cursor": cursor,
            }
            params.update(filter_base_params)

            if self.api_key:
                params["api_key"] = self.api_key
            elif self.mailto:
                params["mailto"] = self.mailto

            prepared_url = requests.Request("GET", OPENALEX_WORKS_URL, headers=headers, params=params).prepare().url
            logger.info(f"Querying OpenAlex Works URL: {prepared_url}")

            max_retries = 5
            response = None
            for attempt in range(max_retries):
                self.enforce_pacing()
                try:
                    response = self.session.get(OPENALEX_WORKS_URL, headers=headers, params=params, timeout=30)
                    if response.status_code == 429:
                        sleep_time = 0
                        retry_after_hdr = response.headers.get("retry-after")
                        if retry_after_hdr and retry_after_hdr.isdigit():
                            sleep_time = float(retry_after_hdr)
                        else:
                            try:
                                body_json = response.json()
                                if "retryAfter" in body_json:
                                    sleep_time = float(body_json["retryAfter"])
                            except Exception:
                                pass
                        
                        if sleep_time > 60:
                            err_msg = (
                                f"OpenAlex daily free credit limit exhausted for today. "
                                f"Resets at midnight UTC (retry after {int(sleep_time)} seconds / ~{sleep_time/3600:.1f} hours)."
                            )
                            logger.error(err_msg, exc_info=True)
                            raise RuntimeError(err_msg)

                        if sleep_time <= 0:
                            sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)

                        logger.warning(
                            f"OpenAlex API rate limited (HTTP 429). Retrying in {sleep_time:.2f}s "
                            f"(Attempt {attempt+1}/{max_retries})..."
                        )
                        time.sleep(sleep_time)
                        continue
                    response.raise_for_status()
                    break
                except requests.HTTPError as http_err:
                    if response is not None and response.status_code == 429 and attempt < max_retries - 1:
                        continue
                    resp_body = response.text if response is not None else "No response body"
                    logger.error(f"HTTP request to OpenAlex API failed for {venue_upper} {year}: {http_err}\nFull Response Body:\n{resp_body}", exc_info=True)
                    raise http_err
                except Exception as e:
                    logger.error(f"Network error querying OpenAlex API for {venue_upper} {year}: {e}", exc_info=True)
                    raise e

            if not initial_request_done and response is not None:
                self.inspect_rate_limit_headers(response)
                initial_request_done = True

            if response is None:
                err_msg = f"Failed to get response from OpenAlex API for {venue_upper} {year}"
                logger.error(err_msg, exc_info=True)
                raise RuntimeError(err_msg)

            data = response.json()
            results = data.get("results", [])
            meta = data.get("meta", {})
            next_cursor = meta.get("next_cursor")

            is_last_page = (not next_cursor) or (len(results) < 200)

            page_papers: List[Dict[str, Any]] = []
            for item in results:
                paper_id = item.get("id") or item.get("doi") or item.get("title")
                doi = item.get("doi")
                title = item.get("display_name") or item.get("title") or ""

                raw_affiliations: List[str] = []
                authorships = item.get("authorships", [])

                for auth in authorships:
                    raw_str = auth.get("raw_affiliation_string")
                    if raw_str and raw_str.strip():
                        raw_affiliations.append(raw_str.strip())
                    
                    affs = auth.get("affiliations", [])
                    for aff_obj in affs:
                        a_str = aff_obj.get("raw_affiliation_string")
                        if a_str and a_str.strip():
                            raw_affiliations.append(a_str.strip())

                    institutions = auth.get("institutions", [])
                    for inst in institutions:
                        display_name = inst.get("display_name")
                        if display_name and display_name.strip():
                            raw_affiliations.append(display_name.strip())

                page_papers.append({
                    "paper_id": str(paper_id),
                    "doi": doi,
                    "title": title,
                    "raw_affiliations": raw_affiliations,
                })

            self.append_raw_batch(
                venue=venue_upper,
                year=year,
                new_papers=page_papers,
                completed=is_last_page,
                next_cursor=next_cursor if not is_last_page else None,
            )

            if is_last_page:
                break

            cursor = next_cursor

        return self.load_cached(venue_upper, year)
