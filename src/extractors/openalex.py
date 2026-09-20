import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

from src.config import (
    OPENALEX_API_KEY,
    OPENALEX_MAILTO,
    OPENALEX_SOURCES_CACHE_PATH,
)
from src.extractors.base import BaseExtractor
from src.normalizer.gemini_client import GeminiClient
from src.utils import sanitize_venue_name

logger = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_SOURCES_URL = "https://api.openalex.org/sources"


def normalize_cache_entry(entry: Any) -> Dict[str, Any]:
    """
    Normalizes cache entry into standard multi-ID / multi-DOI structure with frequency:
    {
        "source_ids": ["S4306506823", ...],
        "doi_prefixes": ["10.1109/icra", ...],
        "frequency": "annual",
        "years": {
            "2023": {
                "source_ids": [...],
                "doi_prefixes": [...]
            }
        }
    }
    Supports legacy string or list cache entries for 100% backward compatibility.
    """
    if isinstance(entry, str):
        clean = entry.split("/")[-1].strip()
        return {"source_ids": [clean] if clean else [], "doi_prefixes": [], "frequency": "annual", "years": {}}
    if isinstance(entry, list):
        clean_list = [str(x).split("/")[-1].strip() for x in entry if str(x).strip()]
        return {"source_ids": clean_list, "doi_prefixes": [], "frequency": "annual", "years": {}}
    if isinstance(entry, dict):
        s_ids = entry.get("source_ids") or []
        if isinstance(s_ids, str):
            s_ids = [s_ids]
        clean_s_ids = [str(x).split("/")[-1].strip() for x in s_ids if str(x).strip()]

        d_prefs = entry.get("doi_prefixes") or []
        if isinstance(d_prefs, str):
            d_prefs = [d_prefs]
        clean_d_prefs = [str(x).strip().lower() for x in d_prefs if str(x).strip()]

        # Handle legacy "source_id" or "doi_prefix" keys
        if not clean_s_ids and entry.get("source_id"):
            clean_s_ids = [str(entry["source_id"]).split("/")[-1].strip()]
        if not clean_d_prefs and entry.get("doi_prefix"):
            clean_d_prefs = [str(entry["doi_prefix"]).strip().lower()]

        freq = str(entry.get("frequency", "annual")).strip().lower()

        years_raw = entry.get("years") or {}
        norm_years = {}
        if isinstance(years_raw, dict):
            for y_k, y_v in years_raw.items():
                norm_years[str(y_k)] = normalize_cache_entry(y_v)

        return {"source_ids": clean_s_ids, "doi_prefixes": clean_d_prefs, "frequency": freq, "years": norm_years}

    return {"source_ids": [], "doi_prefixes": [], "frequency": "annual", "years": {}}


class OpenAlexExtractor(BaseExtractor):
    """
    OpenAlex REST API extractor implementation.
    Harvests publication metadata and institutional affiliations using exact Source IDs or DOI prefixes.
    - Resolves OpenAlex Source IDs and DOI prefixes via Sources API + sample DOIs + Gemini LLM if un-cached.
    - Caches source IDs and DOI prefixes (with support for multiple IDs/DOIs and per-year overrides) in data/openalex_sources_cache.json.
    - Queries Works API using primary_location.source.id or doi_starts_with (never unconstrained raw text search).
    - Supports deterministic pause-and-resume via cursor tokens.
    """

    def __init__(
        self,
        api_key: str = OPENALEX_API_KEY,
        mailto: str = OPENALEX_MAILTO,
        sources_cache_path: Path = OPENALEX_SOURCES_CACHE_PATH,
        model: Optional[str] = None,
        gemini_client: Optional[GeminiClient] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.mailto = mailto
        self.sources_cache_path = sources_cache_path
        if gemini_client:
            self.gemini_client = gemini_client
            if model:
                self.gemini_client.model = model
        else:
            self.gemini_client = GeminiClient(model=model or DEFAULT_GEMINI_MODEL)
        self.rate_limit_delay_seconds = 0.15

    def _load_sources_cache(self) -> Dict[str, Any]:
        """Loads cached venue -> OpenAlex Source info map."""
        if not self.sources_cache_path.exists() or self.sources_cache_path.stat().st_size == 0:
            return {}
        try:
            logger.info(f"Opening file for reading OpenAlex sources cache: {self.sources_cache_path.resolve()}")
            with open(self.sources_cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_sources_cache(self, cache: Dict[str, Any]) -> None:
        """Saves venue -> OpenAlex Source info map to disk."""
        try:
            self.sources_cache_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Opening file for writing OpenAlex sources cache: {self.sources_cache_path.resolve()}")
            with open(self.sources_cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed writing OpenAlex sources cache file: {e}")

    def _get_cached_source_info(
        self,
        cache: Dict[str, Any],
        venue_key: str,
        year: Optional[int] = None,
    ) -> Tuple[List[str], List[str], str]:
        """Retrieves cached source_ids, doi_prefixes, and frequency for a venue and optional year."""
        if venue_key not in cache:
            return [], [], "annual"

        norm_entry = normalize_cache_entry(cache[venue_key])
        freq = norm_entry.get("frequency", "annual")
        if year is not None and "years" in norm_entry and str(year) in norm_entry["years"]:
            y_entry = norm_entry["years"][str(year)]
            y_ids = y_entry.get("source_ids", [])
            y_prefs = y_entry.get("doi_prefixes", [])
            if y_ids or y_prefs:
                eff_ids = y_ids if y_ids else norm_entry.get("source_ids", [])
                eff_prefs = y_prefs if y_prefs else norm_entry.get("doi_prefixes", [])
                return eff_ids, eff_prefs, freq

        return norm_entry.get("source_ids", []), norm_entry.get("doi_prefixes", []), freq

    def _update_sources_cache(
        self,
        cache: Dict[str, Any],
        venue_key: str,
        source_ids: List[str],
        doi_prefixes: List[str],
        frequency: str = "annual",
        year: Optional[int] = None,
    ) -> None:
        """Updates and persists source_ids, doi_prefixes, and frequency in cache for a venue and optional year."""
        curr_entry = normalize_cache_entry(cache.get(venue_key, {}))
        curr_entry["frequency"] = frequency

        if year is not None:
            if "years" not in curr_entry:
                curr_entry["years"] = {}
            curr_entry["years"][str(year)] = {
                "source_ids": source_ids,
                "doi_prefixes": doi_prefixes,
            }

        for s in source_ids:
            if s not in curr_entry["source_ids"]:
                curr_entry["source_ids"].append(s)
        for d in doi_prefixes:
            if d not in curr_entry["doi_prefixes"]:
                curr_entry["doi_prefixes"].append(d)

        cache[venue_key] = curr_entry
        self._save_sources_cache(cache)

    def _fetch_sample_doi_prefixes(self, source_id: str, headers: Dict[str, str]) -> List[str]:
        """Queries OpenAlex Works API for up to 3 sample works of a candidate source ID to extract actual DOI prefixes."""
        clean_id = source_id.split("/")[-1]
        params: Dict[str, Any] = {
            "filter": f"primary_location.source.id:{clean_id}",
            "per-page": 3,
            "select": "doi",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        elif self.mailto:
            params["mailto"] = self.mailto

        prefixes = set()
        try:
            self.enforce_pacing()
            res = self.session.get(OPENALEX_WORKS_URL, headers=headers, params=params, timeout=15)
            if res.status_code == 200:
                works = res.json().get("results", [])
                for w in works:
                    doi_url = w.get("doi") or ""
                    if doi_url:
                        clean_doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
                        parts = clean_doi.split("/")
                        if len(parts) >= 2:
                            prefix = f"{parts[0]}/{parts[1]}".lower()
                            prefixes.add(prefix)
        except Exception as e:
            logger.debug(f"Could not fetch sample DOI prefix for source {clean_id}: {e}")
        return list(prefixes)

    def resolve_source_info(
        self,
        venue: str,
        year: Optional[int] = None,
        search_term: Optional[str] = None,
        openalex_source_id: Optional[Any] = None,
        doi_prefix: Optional[Any] = None,
    ) -> Tuple[List[str], List[str], str]:
        """
        Resolves OpenAlex Source IDs, DOI prefixes, and venue publication frequency for a given venue and optional year.
        Uses explicit parameters if provided; otherwise checks local cache or queries OpenAlex Sources API + sample DOIs + Gemini LLM.
        """
        venue_key = sanitize_venue_name(venue)
        cache = self._load_sources_cache()

        # Parse explicit parameters
        explicit_source_ids = []
        if openalex_source_id:
            if isinstance(openalex_source_id, str):
                explicit_source_ids = [s.strip().split("/")[-1] for s in openalex_source_id.split(",") if s.strip()]
            elif isinstance(openalex_source_id, list):
                explicit_source_ids = [str(s).strip().split("/")[-1] for s in openalex_source_id if str(s).strip()]

        explicit_doi_prefixes = []
        if doi_prefix:
            if isinstance(doi_prefix, str):
                explicit_doi_prefixes = [d.strip().lower() for d in doi_prefix.split(",") if d.strip()]
            elif isinstance(doi_prefix, list):
                explicit_doi_prefixes = [str(d).strip().lower() for d in doi_prefix if str(d).strip()]

        if explicit_source_ids or explicit_doi_prefixes:
            logger.info(
                f"Using explicit configuration for venue '{venue_key}': "
                f"source_ids={explicit_source_ids}, doi_prefixes={explicit_doi_prefixes}"
            )
            self._update_sources_cache(cache, venue_key, explicit_source_ids, explicit_doi_prefixes, frequency="annual", year=year)
            return explicit_source_ids, explicit_doi_prefixes, "annual"

        # Check local cache
        cached_ids, cached_prefs, cached_freq = self._get_cached_source_info(cache, venue_key, year=year)
        if cached_ids or cached_prefs:
            logger.info(
                f"Using cached OpenAlex source info for venue '{venue_key}' (year={year}): "
                f"source_ids={cached_ids}, doi_prefixes={cached_prefs}, frequency={cached_freq}"
            )
            return cached_ids, cached_prefs, cached_freq

        # Construct list of search terms
        search_terms_list: List[str] = []
        if isinstance(search_term, list):
            search_terms_list = [str(s).strip() for s in search_term if str(s).strip()]
        elif isinstance(search_term, str) and search_term.strip():
            search_terms_list = [search_term.strip()]

        if not search_terms_list:
            search_terms_list = [venue]

        headers: Dict[str, str] = {}
        if self.api_key:
            headers["api-key"] = self.api_key

        # Collect and deduplicate candidates across ALL search terms
        all_candidates_by_id: Dict[str, Dict[str, Any]] = {}
        for st_term in search_terms_list:
            logger.info(f"Resolving OpenAlex Source IDs & DOIs for venue '{venue_key}' using search term '{st_term}' via Sources API...")
            params: Dict[str, Any] = {"search": st_term}
            if self.api_key:
                params["api_key"] = self.api_key
            elif self.mailto:
                params["mailto"] = self.mailto

            try:
                self.enforce_pacing()
                res = self.session.get(OPENALEX_SOURCES_URL, headers=headers, params=params, timeout=30)
                if res.status_code == 200:
                    results = res.json().get("results", [])
                    for cand in results:
                        cand_id = cand.get("id", "").split("/")[-1]
                        if cand_id and cand_id not in all_candidates_by_id:
                            all_candidates_by_id[cand_id] = cand
            except Exception as e:
                logger.warning(f"Error querying OpenAlex Sources API for search term '{st_term}': {e}")

        raw_results = list(all_candidates_by_id.values())
        if not raw_results:
            err_msg = f"No OpenAlex sources found for search terms {search_terms_list}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        valid_results = [c for c in raw_results if (c.get("works_count") or 0) > 0]
        if not valid_results:
            valid_results = raw_results

        # Fetch sample DOI prefixes for ALL candidate sources with works_count > 0
        for cand in valid_results:
            cand_id = cand.get("id", "").split("/")[-1]
            if cand_id and (cand.get("works_count") or 0) > 0:
                sample_prefs = self._fetch_sample_doi_prefixes(cand_id, headers)
                cand["sample_doi_prefixes"] = sample_prefs
            else:
                cand["sample_doi_prefixes"] = []

        logger.info(f"Sending ALL {len(valid_results)} unique candidate OpenAlex sources (collected across search terms {search_terms_list}) to Gemini LLM for evaluation...")
        gemini_res = self.gemini_client.resolve_openalex_venue_sources(
            venue_key,
            valid_results,
            short_name=venue_key,
            search_term=search_terms_list if len(search_terms_list) > 1 else search_terms_list[0],
        )
        resolved_ids = gemini_res.get("source_ids", [])
        resolved_prefs = gemini_res.get("doi_prefixes", [])
        resolved_freq = gemini_res.get("frequency", "annual")

        if not resolved_ids and valid_results:
            best_cand = max(valid_results, key=lambda x: x.get("works_count", 0))
            best_id = best_cand.get("id", "").split("/")[-1]
            if best_id:
                resolved_ids = [best_id]
                resolved_prefs = best_cand.get("sample_doi_prefixes", [])
                logger.info(f"Fallback selected source ID '{best_id}' ({best_cand.get('display_name')}) with max works_count={best_cand.get('works_count')}")

        self._update_sources_cache(cache, venue_key, resolved_ids, resolved_prefs, frequency=resolved_freq, year=year)
        logger.info(
            f"Successfully cached OpenAlex source info for venue '{venue_key}' (year={year}): "
            f"source_ids={resolved_ids}, doi_prefixes={resolved_prefs}, frequency={resolved_freq}"
        )
        return resolved_ids, resolved_prefs, resolved_freq

    def resolve_source_id(
        self,
        venue: str,
        search_term: Optional[str] = None,
        openalex_source_id: Optional[str] = None,
    ) -> str:
        """Backward compatible wrapper around resolve_source_info."""
        s_ids, _, _ = self.resolve_source_info(
            venue, search_term=search_term, openalex_source_id=openalex_source_id
        )
        return s_ids[0] if s_ids else ""

    def _determine_filter_params(
        self,
        venue_upper: str,
        year: int,
        source_ids: List[str],
        doi_prefixes: List[str],
        headers: Dict[str, str],
        search_term: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Determines optimal filter parameters for OpenAlex Works API.
        Tries primary_location.source.id (supporting multi-ID OR) first.
        If count is 0, tries doi_starts_with (supporting candidate DOI prefixes resolved via OpenAlex & Gemini).
        STRICT REQUIREMENT: Never falls back to unconstrained raw text search ('search: {venue}').
        """
        # 1. Try primary_location.source.id
        if source_ids:
            source_filter = f"publication_year:{year},primary_location.source.id:{'|'.join(source_ids)}"
            test_params: Dict[str, Any] = {
                "filter": source_filter,
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
                        logger.info(f"OpenAlex filter strategy for {venue_upper} {year}: {source_filter} (count={cnt})")
                        return {"filter": source_filter}
            except Exception as e:
                logger.warning(f"Failed testing primary_location.source.id filter for {venue_upper} {year}: {e}")

        # 2. Try doi_starts_with
        if doi_prefixes:
            doi_filter = f"publication_year:{year},doi_starts_with:{'|'.join(doi_prefixes)}"
            test_params = {
                "filter": doi_filter,
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
                        logger.info(f"OpenAlex filter strategy for {venue_upper} {year}: {doi_filter} (count={cnt})")
                        return {"filter": doi_filter}
            except Exception as e:
                logger.warning(f"Failed testing doi_starts_with filter for {venue_upper} {year}: {e}")

        # 3. Fallback filter (count=0)
        if doi_prefixes:
            fallback_filter = f"publication_year:{year},doi_starts_with:{'|'.join(doi_prefixes)}"
        elif source_ids:
            fallback_filter = f"publication_year:{year},primary_location.source.id:{'|'.join(source_ids)}"
        else:
            fallback_filter = f"publication_year:{year}"

        logger.warning(
            f"No OpenAlex works found for {venue_upper} {year} with source IDs '{source_ids}' or DOI prefixes '{doi_prefixes}'. "
            f"Using filter '{fallback_filter}' (count=0) to prevent unconstrained raw search corruption."
        )
        return {"filter": fallback_filter}

    def extract(
        self,
        venue: str,
        year: int,
        force: bool = False,
        search_term: Optional[str] = None,
        openalex_source_id: Optional[Any] = None,
        doi_prefix: Optional[Any] = None,
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

        source_ids, doi_prefixes, frequency = self.resolve_source_info(
            venue_upper,
            year=year,
            search_term=search_term,
            openalex_source_id=openalex_source_id,
            doi_prefix=doi_prefix,
        )

        # Off-year check for biennial conferences
        is_off_year = False
        if frequency == "biennial_even" and (year % 2 != 0):
            is_off_year = True
        elif frequency == "biennial_odd" and (year % 2 == 0):
            is_off_year = True

        if is_off_year:
            logger.info(
                f"Venue '{venue_upper}' is identified as a biennial conference ({frequency}). "
                f"Year {year} is an off-year. Skipping OpenAlex API harvesting and marking dataset completed with 0 papers."
            )
            return self.append_raw_batch(venue_upper, year, [], completed=True, next_cursor=None)

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
            venue_upper, year, source_ids, doi_prefixes, headers, search_term=search_term
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
