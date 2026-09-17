import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from src.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)


class ConferenceScheduleExtractor(BaseExtractor):
    """
    Conference Schedule HTML/JSON Extractor implementation.
    Parses official online conference programs or schedule pages (e.g. Papercept, epapers)
    when official publisher indexes are delayed (e.g., ICRA 2026).
    Uses BeautifulSoup to parse session paper cards and author affiliation blocks.
    Strictly raises exceptions if fetching or parsing fails.
    """

    def __init__(self, schedule_url: Optional[str] = None, schedule_file: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self.schedule_url = schedule_url
        self.schedule_file = schedule_file

    def _parse_html_schedule(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Parses conference schedule HTML structure using BeautifulSoup.
        Extracts paper cards, paper IDs, titles, and author affiliation blocks.
        """
        if BeautifulSoup is None:
            logger.warning("beautifulsoup4 is not installed; falling back to regex HTML parser.")
            import re
            papers = []
            paper_blocks = re.findall(r'<div[^>]*class="[^"]*paper[^"]*"[^>]*>(.*?)</div>\s*</div>', html_content, re.DOTALL | re.IGNORECASE)
            if not paper_blocks:
                paper_blocks = re.findall(r'<div[^>]*id="PAPER-\d+"[^>]*>(.*?)</div>\s*</div>', html_content, re.DOTALL | re.IGNORECASE)
            
            for idx, block in enumerate(paper_blocks, 1):
                t_match = re.search(r'<h[34][^>]*>(.*?)</h[34]>', block, re.DOTALL | re.IGNORECASE)
                title = t_match.group(1).strip() if t_match else f"Paper {idx}"
                id_match = re.search(r'id="([^"]+)"', block)
                paper_id = id_match.group(1) if id_match else f"SCHED-{idx:04d}"
                affs = re.findall(r'class="[^"]*affil[^"]*"[^>]*>(.*?)</span>', block, re.DOTALL | re.IGNORECASE)
                clean_affs = [re.sub(r'<[^>]+>', '', a).strip() for a in affs if a.strip()]
                papers.append({
                    "paper_id": paper_id,
                    "title": title,
                    "raw_affiliations": clean_affs,
                })
            return papers

        soup = BeautifulSoup(html_content, "html.parser")
        papers: List[Dict[str, Any]] = []

        # Find paper cards/rows (supports common conference schedule CSS classes/tags)
        paper_blocks = soup.find_all(class_=lambda c: c and any(k in c.lower() for k in ["paper", "session-item", "presentation", "slot"]))

        if not paper_blocks:
            # Fallback: search table rows or divs containing paper title elements
            paper_blocks = soup.find_all(["tr", "div"], class_=True)

        paper_index = 1
        for block in paper_blocks:
            title_elem = block.find(class_=lambda c: c and "title" in c.lower()) or block.find(["h3", "h4", "strong", "a"])
            if not title_elem:
                continue

            title = title_elem.get_text(strip=True)
            if not title or len(title) < 5:
                continue

            # Paper ID
            id_attr = block.get("id") or block.get("data-paper-id")
            if id_attr:
                paper_id = str(id_attr)
            else:
                paper_id = f"SCHED-{paper_index:04d}"
                paper_index += 1

            # Extract author affiliations
            raw_affiliations: List[str] = []
            affil_elems = block.find_all(class_=lambda c: c and any(k in c.lower() for k in ["affil", "author", "org", "institution"]))
            
            for elem in affil_elems:
                text = elem.get_text(strip=True)
                if text and len(text) > 2:
                    raw_affiliations.append(text)

            papers.append({
                "paper_id": paper_id,
                "title": title,
                "raw_affiliations": raw_affiliations,
            })

        return papers

    def extract(self, venue: str, year: int, force: bool = False) -> Dict[str, Any]:
        venue_upper = venue.upper()
        if not force and self.is_cached(venue_upper, year):
            logger.info(f"Raw data for {venue_upper} {year} is cached.")
            return self.load_cached(venue_upper, year)

        logger.info(f"Extracting schedule data for venue '{venue_upper}' year {year}")

        html_content = ""
        if self.schedule_file:
            path = Path(self.schedule_file)
            if not path.exists():
                err_msg = f"Specified schedule file does not exist: {path}"
                logger.error(err_msg, exc_info=True)
                raise FileNotFoundError(err_msg)
            try:
                html_content = path.read_text(encoding="utf-8")
            except Exception as e:
                logger.error(f"Failed to read schedule HTML file at {path}", exc_info=True)
                raise e
        elif self.schedule_url:
            self.enforce_pacing()
            try:
                response = self.session.get(self.schedule_url, timeout=30)
                response.raise_for_status()
                self.inspect_rate_limit_headers(response)
                html_content = response.text
            except Exception as e:
                logger.error(f"Failed to fetch conference schedule from URL {self.schedule_url}", exc_info=True)
                raise e
        else:
            err_msg = "Neither schedule_url nor schedule_file was provided for ConferenceScheduleExtractor."
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        papers = self._parse_html_schedule(html_content)
        if not papers:
            err_msg = f"Failed to parse any paper records from schedule content for {venue_upper} {year}."
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        return self.save_raw_artifact(venue_upper, year, papers)
