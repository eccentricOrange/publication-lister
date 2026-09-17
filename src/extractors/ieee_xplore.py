import logging
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
    Strictly raises exceptions if API key is missing or network requests fail.
    """

    def __init__(self, api_key: str = IEEE_API_KEY, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key

    def extract(self, venue: str, year: int, force: bool = False) -> Dict[str, Any]:
        venue_upper = venue.upper()
        if not force and self.is_cached(venue_upper, year):
            logger.info(f"Raw data for {venue_upper} {year} is cached.")
            return self.load_cached(venue_upper, year)

        if not self.api_key:
            err_msg = "IEEE_API_KEY is not set in environment or configuration. Cannot proceed with IEEE extraction."
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        pub_title = VENUE_QUERY_MAP.get(venue_upper, venue)
        logger.info(f"Querying IEEE Xplore API for venue '{venue_upper}' ('{pub_title}') year {year}")

        papers: List[Dict[str, Any]] = []
        page_number = 1
        max_records = 200
        total_records = 1
        initial_request_done = False

        while len(papers) < total_records:
            params = {
                "apikey": self.api_key,
                "publication_title": pub_title,
                "publication_year": year,
                "max_records": max_records,
                "start_record": (page_number - 1) * max_records + 1,
                "format": "json",
            }

            self.enforce_pacing()

            try:
                response = self.session.get(IEEE_API_URL, params=params, timeout=30)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"HTTP request to IEEE Xplore API failed for {venue_upper} {year}: {e}", exc_info=True)
                raise e

            if not initial_request_done:
                self.inspect_rate_limit_headers(response)
                initial_request_done = True

            data = response.json()
            total_records = int(data.get("total_records", 0))
            articles = data.get("articles", [])

            if not articles:
                logger.warning(f"No articles returned by IEEE Xplore for {venue_upper} {year}")
                break

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

                papers.append({
                    "paper_id": paper_id,
                    "title": title,
                    "raw_affiliations": raw_affiliations,
                })

            page_number += 1
            if total_records == 0:
                break

        return self.save_raw_artifact(venue_upper, year, papers)

