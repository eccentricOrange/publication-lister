import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import requests

from src.config import SCOPUS_API_KEY
from src.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
SCOPUS_MAX_RESULT_LIMIT = 5000  # Hard Elsevier API limit for deep pagination start offset


class ScopusExtractor(BaseExtractor):
    """
    Elsevier Scopus Search API extractor implementation.
    Retrieves publication entries and author affiliations with concurrent pagination,
    automatic exponential backoff rate limit retries, and full pause/resume support.
    Exits immediately on the first unhandled error.
    """

    def __init__(self, api_key: str = SCOPUS_API_KEY, max_workers: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.max_workers = max_workers

    def _fetch_url_with_retry(
        self,
        headers: Dict[str, str],
        params: Dict[str, Any],
        max_retries: int = 5,
    ) -> Dict[str, Any]:
        """
        Executes Scopus HTTP GET request with rate-limit detection and exponential backoff retries.
        If all retries fail or a non-retryable error occurs, raises exception immediately.
        """
        last_exception = None

        for attempt in range(1, max_retries + 1):
            self.enforce_pacing()
            try:
                response = self.session.get(SCOPUS_SEARCH_URL, headers=headers, params=params, timeout=30)

                # Check for rate limits (429) or transient server errors (500, 502, 503, 504)
                if response.status_code in (429, 500, 502, 503, 504):
                    retry_after = response.headers.get("Retry-After") or response.headers.get("X-RateLimit-Reset")
                    cooldown = (2 ** (attempt - 1)) * 2.0 + random.uniform(0.5, 1.5)
                    if retry_after:
                        try:
                            val = float(retry_after)
                            if val > time.time():
                                cooldown = val - time.time() + 1.0
                            elif 0 < val < 3600:
                                cooldown = val
                        except ValueError:
                            pass

                    logger.warning(
                        f"Scopus API rate limited / server busy ({response.status_code}). "
                        f"Cooling off for {cooldown:.2f}s (Attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(cooldown)
                    continue

                response.raise_for_status()
                return response.json()

            except Exception as e:
                last_exception = e
                # Check for non-retryable 400 Bad Request error (e.g. invalid query or parameter)
                if getattr(e, "response", None) is not None and e.response.status_code == 400:
                    resp_body = e.response.text
                    logger.error(
                        f"Scopus API non-retryable 400 Bad Request error for params={params}: {e}\nResponse Body:\n{resp_body}",
                        exc_info=True,
                    )
                    raise e

                if attempt < max_retries:
                    cooldown = (2 ** (attempt - 1)) * 2.0 + random.uniform(0.5, 1.5)
                    logger.warning(f"Request error querying Scopus API ({e}). Retrying in {cooldown:.2f}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(cooldown)
                else:
                    resp_body = getattr(e, "response", None)
                    text_body = resp_body.text if resp_body is not None else "No response body"
                    logger.error(
                        f"Exhausted retries querying Scopus API for params={params}: {e}\nResponse Body:\n{text_body}",
                        exc_info=True,
                    )
                    raise e

        if last_exception:
            raise last_exception
        raise RuntimeError(f"Scopus API request failed after {max_retries} attempts for params={params}")

    def extract(
        self,
        venue: str,
        year: int,
        force: bool = False,
        query_term: Optional[Any] = None,
        search_term: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            err_msg = "SCOPUS_API_KEY is not set in environment or configuration. Cannot proceed with Scopus extraction."
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        venue_upper = venue.upper()

        # Pause/Resume check
        state = self.get_resume_state(venue_upper, year)
        if not force and state.get("completed"):
            logger.info(f"Raw data for {venue_upper} {year} is cached and complete.")
            return self.load_cached(venue_upper, year)

        start_index = state.get("next_page", 0) if not force else 0
        if not isinstance(start_index, int):
            start_index = 0

        logger.info(f"Querying Elsevier Scopus API for venue '{venue_upper}' year {year} (start offset {start_index}, max_workers={self.max_workers})")

        headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/json",
        }

        # Build search query clause from search_term, query_term, or venue
        terms_to_use: List[str] = []
        eff_term = search_term or query_term
        if isinstance(eff_term, str):
            terms_to_use = [eff_term]
        elif isinstance(eff_term, list):
            terms_to_use = [str(t).strip() for t in eff_term if str(t).strip()]

        if not terms_to_use:
            terms_to_use = [venue]

        or_clauses = [f'SRCTITLE("{term}")' for term in terms_to_use]
        terms_clause = " OR ".join(or_clauses)
        if len(or_clauses) > 1:
            terms_clause = f"({terms_clause})"

        query_str = f"{terms_clause} AND PUBYEAR IS {year}"
        count_per_page = 25

        # 1. Initial request to discover total_results and rate limits
        params_init = {
            "query": query_str,
            "start": start_index,
            "count": count_per_page,
        }

        data_init = self._fetch_url_with_retry(headers, params_init)
        search_results = data_init.get("search-results", {})
        total_results = int(search_results.get("opensearch:totalResults", 0))
        entries_init = search_results.get("entry", [])

        if not entries_init:
            logger.warning(f"No entries returned by Scopus for {venue_upper} {year} at start={start_index}")
            return self.append_raw_batch(venue_upper, year, [], next_page=start_index, completed=True)

        def parse_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
            paper_id = entry.get("dc:identifier") or entry.get("prism:doi") or entry.get("dc:title")
            title = entry.get("dc:title", "")

            raw_affiliations: List[str] = []
            affil_list = entry.get("affiliation", [])
            if isinstance(affil_list, list):
                for aff in affil_list:
                    aff_name = aff.get("affilname") or aff.get("affiliation-city")
                    if aff_name:
                        raw_affiliations.append(aff_name.strip())
            elif isinstance(affil_list, dict):
                aff_name = affil_list.get("affilname")
                if aff_name:
                    raw_affiliations.append(aff_name.strip())

            return {
                "paper_id": str(paper_id),
                "title": title,
                "raw_affiliations": raw_affiliations,
            }

        max_retrievable = min(total_results, SCOPUS_MAX_RESULT_LIMIT)
        if total_results > SCOPUS_MAX_RESULT_LIMIT:
            logger.warning(
                f"Scopus API search results ({total_results} total papers) exceed Elsevier Scopus "
                f"maximum retrieval depth limit ({SCOPUS_MAX_RESULT_LIMIT} results). "
                f"Capping retrieval at {SCOPUS_MAX_RESULT_LIMIT} papers for {venue_upper} {year}."
            )

        batch_0 = [parse_entry(e) for e in entries_init]
        next_idx = start_index + count_per_page
        self.append_raw_batch(
            venue_upper,
            year,
            batch_0,
            next_page=next_idx,
            completed=next_idx >= max_retrievable,
        )

        if next_idx >= max_retrievable:
            return self.load_cached(venue_upper, year)

        # 2. Concurrent fetching for remaining page offsets up to max_retrievable (5000)
        remaining_offsets = list(range(next_idx, max_retrievable, count_per_page))
        logger.info(f"Fetching remaining {len(remaining_offsets)} pages ({max_retrievable}/{total_results} total papers) concurrently with max_workers={self.max_workers}...")

        def fetch_offset(offset: int) -> Tuple[int, List[Dict[str, Any]]]:
            p = {
                "query": query_str,
                "start": offset,
                "count": count_per_page,
            }
            d = self._fetch_url_with_retry(headers, p)
            ent = d.get("search-results", {}).get("entry", [])
            parsed = [parse_entry(e) for e in ent]
            return offset, parsed

        fetched_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_offset = {executor.submit(fetch_offset, off): off for off in remaining_offsets}
            for future in as_completed(future_to_offset):
                # Calling future.result() without try/except ensures any unhandled error exits immediately on the first error!
                offset, page_papers = future.result()
                fetched_count += 1
                is_done = fetched_count == len(remaining_offsets)
                self.append_raw_batch(
                    venue_upper,
                    year,
                    page_papers,
                    next_page=offset + count_per_page,
                    completed=is_done,
                )

        return self.save_raw_artifact(
            venue_upper,
            year,
            self.get_resume_state(venue_upper, year).get("papers", []),
            completed=True,
        )

