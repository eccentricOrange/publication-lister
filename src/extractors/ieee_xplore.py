import logging
import random
import time
from typing import Any, Dict, List, Optional
import requests

from src.config import IEEE_API_KEY
from src.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

IEEE_API_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"


class IEEEExtractor(BaseExtractor):
    """
    IEEE Xplore REST API extractor implementation.
    Retrieves publication metadata and raw author affiliation strings.
    Strictly enforces:
      1. Maximum 1 request per second pacing delay.
      2. Maximum 200 records per query pagination.
      3. Automatic retry & exponential backoff on HTTP 403 & 429 errors.
      4. Deterministic pause-and-resume pagination state.
    Strictly raises exceptions if API key is missing or max retries exhausted.
    """

    def __init__(self, api_key: str = IEEE_API_KEY, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        # Enforce no more than 1 request per second
        self.rate_limit_delay_seconds = 1.0

    def extract(self, venue: str, year: int, force: bool = False, query_term: Optional[str] = None) -> Dict[str, Any]:
        if not self.api_key:
            err_msg = "IEEE_API_KEY is not set in environment or configuration. Cannot proceed with IEEE extraction."
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        venue_upper = venue.upper()
        raw_path = self.get_raw_file_path(venue_upper, year)

        if not force and self.is_cached(venue_upper, year):
            logger.info(f"Raw data for {venue_upper} {year} is fully cached and completed.")
            return self.load_cached(venue_upper, year)

        page_number = 1
        max_records = 200  # Strict cap: 200 results per query
        existing_papers_count = 0

        # Pause/Resume handling: check if partial artifact exists on disk
        if not force and raw_path.exists() and raw_path.stat().st_size > 0:
            try:
                cached_data = self.load_cached(venue_upper, year)
                if not cached_data.get("completed", False):
                    existing_papers_count = cached_data.get("total_papers", 0)
                    page_number = (existing_papers_count // max_records) + 1
                    logger.info(f"Resuming IEEE Xplore extraction for {venue_upper} {year} from paper #{existing_papers_count + 1} (page {page_number})")
            except Exception:
                page_number = 1

        pub_title = query_term or venue_upper
        logger.info(f"Querying IEEE Xplore API for venue '{venue_upper}' ('{pub_title}') year {year}")

        total_records = 1
        initial_request_done = False

        while True:
            start_record = (page_number - 1) * max_records + 1
            if existing_papers_count > 0 and start_record <= existing_papers_count:
                # Align start_record to next un-fetched page
                start_record = existing_papers_count + 1

            params = {
                "apikey": self.api_key,
                "publication_title": pub_title,
                "publication_year": year,
                "max_records": max_records,
                "start_record": start_record,
                "format": "json",
            }

            prepared_url = requests.Request("GET", IEEE_API_URL, params=params).prepare().url
            logger.info(f"Querying IEEE Xplore URL: {prepared_url}")

            max_retries = 5
            response = None
            for attempt in range(max_retries):
                self.enforce_pacing()
                try:
                    response = self.session.get(IEEE_API_URL, params=params, timeout=30)
                    if response.status_code in (403, 429):
                        retry_after = response.headers.get("retry-after")
                        sleep_time = float(retry_after) if retry_after else (2 ** attempt) * 2.0 + random.uniform(0.5, 1.5)
                        logger.warning(f"IEEE API rate limited ({response.status_code}). Sleeping {sleep_time:.2f}s (Attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_time)
                        continue
                    response.raise_for_status()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        sleep_time = (2 ** attempt) * 2.0 + random.uniform(0.5, 1.5)
                        logger.warning(f"Request error querying IEEE Xplore ({e}). Retrying in {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Exhausted retries querying IEEE Xplore for {venue_upper} {year}: {e}", exc_info=True)
                        raise e

            if not response:
                raise RuntimeError(f"Failed receiving data from IEEE API for {venue_upper} {year}")

            data = response.json()
            if not initial_request_done:
                total_records = data.get("total_records", 0)
                initial_request_done = True
                logger.info(f"IEEE Xplore total records for {venue_upper} {year}: {total_records}")

            articles = data.get("articles", [])
            paper_batch: List[Dict[str, Any]] = []

            for art in articles:
                paper_id = art.get("article_number") or art.get("doi") or f"IEEE-{page_number}"
                title = art.get("title") or ""
                authors_info = art.get("authors", {}).get("author", [])
                
                raw_affiliations: List[str] = []
                for auth in authors_info:
                    aff = auth.get("affiliation")
                    if aff and str(aff).strip():
                        raw_affiliations.append(str(aff).strip())

                paper_batch.append({
                    "paper_id": str(paper_id),
                    "title": title,
                    "raw_affiliations": raw_affiliations,
                })

            fetched_so_far = (page_number - 1) * max_records + len(paper_batch)
            is_completed = (fetched_so_far >= total_records or len(paper_batch) == 0)

            checkpoint = self.append_raw_batch(
                venue_upper,
                year,
                paper_batch,
                completed=is_completed,
                next_page=page_number + 1,
            )

            logger.info(f"IEEE Xplore Page {page_number}: +{len(paper_batch)} papers (Total: {checkpoint['total_papers']}/{total_records})")

            if is_completed:
                logger.info(f"Successfully completed IEEE Xplore harvesting for {venue_upper} {year} ({checkpoint['total_papers']} total papers).")
                return checkpoint

            page_number += 1
