import logging
import random
import time
from typing import Any, Dict, List, Optional
import requests

from src.config import IEEE_API_KEY
from src.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

IEEE_API_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

# Venue string mapping for IEEE Xplore API query search
VENUE_QUERY_MAP = {
    "ICRA": "IEEE International Conference on Robotics and Automation",
    "IROS": "IEEE/RSJ International Conference on Intelligent Robots and Systems",
    "RA-L": "IEEE Robotics and Automation Letters",
    "TRO": "IEEE Transactions on Robotics",
}


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

    def extract(self, venue: str, year: int, force: bool = False) -> Dict[str, Any]:
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

        pub_title = VENUE_QUERY_MAP.get(venue_upper, venue)
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
                        if retry_after and retry_after.isdigit():
                            sleep_time = float(retry_after)
                        else:
                            sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                        logger.warning(
                            f"IEEE Xplore API rate/access limited (HTTP {response.status_code}). Retrying in {sleep_time:.2f}s "
                            f"(Attempt {attempt+1}/{max_retries})..."
                        )
                        time.sleep(sleep_time)
                        continue
                    response.raise_for_status()
                    break
                except requests.HTTPError as http_err:
                    if response is not None and response.status_code in (403, 429) and attempt < max_retries - 1:
                        continue
                    resp_body = response.text if response is not None else "No response body"
                    logger.error(f"HTTP request to IEEE Xplore API failed for {venue_upper} {year}: {http_err}\nFull Response Body:\n{resp_body}", exc_info=True)
                    raise http_err
                except Exception as e:
                    logger.error(f"Network error querying IEEE Xplore API for {venue_upper} {year}: {e}", exc_info=True)
                    raise e

            if not initial_request_done and response is not None:
                self.inspect_rate_limit_headers(response)
                initial_request_done = True

            if response is None:
                err_msg = f"Failed to get response from IEEE Xplore API for {venue_upper} {year}"
                logger.error(err_msg, exc_info=True)
                raise RuntimeError(err_msg)

            data = response.json()
            total_records = int(data.get("total_records", 0))
            articles = data.get("articles", [])

            if not articles:
                logger.warning(f"No articles returned by IEEE Xplore for {venue_upper} {year}")
                # Mark extraction complete if no more articles returned
                self.append_raw_batch(venue_upper, year, [], completed=True)
                break

            page_papers: List[Dict[str, Any]] = []
            for art in articles:
                paper_id = str(art.get("article_number") or art.get("doi") or art.get("title"))
                title = art.get("title", "")
                authors_data = art.get("authors", {}).get("author", [])

                raw_affiliations: List[str] = []
                if isinstance(authors_data, list):
                    for auth in authors_data:
                        aff = auth.get("affiliation")
                        if aff:
                            raw_affiliations.append(aff.strip())
                elif isinstance(authors_data, dict):
                    aff = authors_data.get("affiliation")
                    if aff:
                        raw_affiliations.append(aff.strip())

                page_papers.append({
                    "paper_id": paper_id,
                    "title": title,
                    "raw_affiliations": raw_affiliations,
                })

            # Incremental disk caching: flush page papers and update resume start_record state on disk
            saved_artifact = self.append_raw_batch(
                venue=venue_upper,
                year=year,
                new_papers=page_papers,
                completed=False,
                next_page=page_number + 1,
            )

            current_total = saved_artifact.get("total_papers", 0)
            if current_total >= total_records or len(articles) < max_records:
                # Mark completed
                self.append_raw_batch(venue_upper, year, [], completed=True)
                break

            page_number += 1

        return self.load_cached(venue_upper, year)
