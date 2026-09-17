import logging
from typing import Any, Dict, List, Optional
import requests

from src.config import SCOPUS_API_KEY
from src.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"


class ScopusExtractor(BaseExtractor):
    """
    Elsevier Scopus Search API extractor implementation.
    Retrieves publication entries and author affiliations.
    Strictly raises exceptions if API key is missing or network requests fail.
    """

    def __init__(self, api_key: str = SCOPUS_API_KEY, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key

    def extract(self, venue: str, year: int, force: bool = False) -> Dict[str, Any]:
        venue_upper = venue.upper()
        if not force and self.is_cached(venue_upper, year):
            logger.info(f"Raw data for {venue_upper} {year} is cached.")
            return self.load_cached(venue_upper, year)

        if not self.api_key:
            err_msg = "SCOPUS_API_KEY is not set in environment or configuration. Cannot proceed with Scopus extraction."
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        logger.info(f"Querying Elsevier Scopus API for venue '{venue_upper}' year {year}")

        headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/json",
        }

        query_str = f"EXACTSRCTITLE({venue}) AND PUBYEAR IS {year}"
        papers: List[Dict[str, Any]] = []
        start_index = 0
        count_per_page = 25
        total_results = 1
        initial_request_done = False

        while start_index < total_results:
            params = {
                "query": query_str,
                "start": start_index,
                "count": count_per_page,
            }

            self.enforce_pacing()

            try:
                response = self.session.get(SCOPUS_SEARCH_URL, headers=headers, params=params, timeout=30)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"HTTP request to Scopus API failed for {venue_upper} {year}: {e}", exc_info=True)
                raise e

            if not initial_request_done:
                self.inspect_rate_limit_headers(response)
                initial_request_done = True

            data = response.json()
            search_results = data.get("search-results", {})
            total_results = int(search_results.get("opensearch:totalResults", 0))
            entries = search_results.get("entry", [])

            if not entries:
                logger.warning(f"No entries returned by Scopus for {venue_upper} {year}")
                break

            for entry in entries:
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

                papers.append({
                    "paper_id": str(paper_id),
                    "title": title,
                    "raw_affiliations": raw_affiliations,
                })

            start_index += count_per_page
            if total_results == 0:
                break

        return self.save_raw_artifact(venue_upper, year, papers)

