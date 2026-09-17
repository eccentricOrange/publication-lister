import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config import NORMALIZED_DATA_DIR, RAW_DATA_DIR
from src.normalizer.gemini_client import GeminiClient
from src.registry.organization_registry import OrganizationRegistry

logger = logging.getLogger(__name__)


class AffiliationNormalizer:
    """
    Normalizes raw paper affiliations against canonical organization registry.
    Uses local registry search first, and queries Gemini API for novel strings.
    Applies within-paper deduplication and updates central canonical registry.
    """

    def __init__(
        self,
        registry: Optional[OrganizationRegistry] = None,
        gemini_client: Optional[GeminiClient] = None,
        raw_dir: Path = RAW_DATA_DIR,
        normalized_dir: Path = NORMALIZED_DATA_DIR,
    ):
        self.registry = registry or OrganizationRegistry()
        self.gemini_client = gemini_client or GeminiClient()
        self.raw_dir = raw_dir
        self.normalized_dir = normalized_dir

    def get_normalized_file_path(self, venue: str, year: int) -> Path:
        """Returns standard normalized path data/normalized/<venue>_<year>_normalized.json."""
        self.normalized_dir.mkdir(parents=True, exist_ok=True)
        return self.normalized_dir / f"{venue.upper()}_{year}_normalized.json"

    def normalize_venue_year(
        self,
        venue: str,
        year: int,
        batch_size: int = 20,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Normalizes raw paper affiliation dataset for a venue and year.
        Dumps output to data/normalized/<venue>_<year>_normalized.json.
        """
        venue_upper = venue.upper()
        norm_path = self.get_normalized_file_path(venue_upper, year)

        if not force and norm_path.exists() and norm_path.stat().st_size > 0:
            logger.info(f"Normalized dataset for {venue_upper} {year} already exists at {norm_path}")
            try:
                with open(norm_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed loading existing normalized dataset at {norm_path}", exc_info=True)
                raise e

        # Load raw file
        raw_file = self.raw_dir / venue_upper / f"{venue_upper}_{year}.json"
        if not raw_file.exists():
            err_msg = f"Raw dataset not found at {raw_file}. Run extract subcommand first."
            logger.error(err_msg, exc_info=True)
            raise FileNotFoundError(err_msg)

        try:
            with open(raw_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load raw paper dataset at {raw_file}", exc_info=True)
            raise e

        raw_papers = raw_data.get("papers", [])
        logger.info(f"Starting normalization for {venue_upper} {year} ({len(raw_papers)} raw papers)")

        # Collect unique raw affiliation strings across all papers
        unique_raw_strings: Set[str] = set()
        for p in raw_papers:
            for aff in p.get("raw_affiliations", []):
                if aff and aff.strip():
                    unique_raw_strings.add(aff.strip())

        # Step 1: Resolve locally using existing registry
        string_to_canonical_id: Dict[str, str] = {}
        unresolved_strings: List[str] = []

        for raw_str in sorted(list(unique_raw_strings)):
            match = self.registry.find_by_string(raw_str)
            if match:
                string_to_canonical_id[raw_str] = match["canonical_id"]
            else:
                unresolved_strings.append(raw_str)

        logger.info(f"Local registry matched {len(string_to_canonical_id)} strings. {len(unresolved_strings)} unresolved strings remaining.")

        # Step 2: Batch unresolved strings and query Gemini API
        if unresolved_strings:
            registry_summary = [
                {
                    "canonical_id": e["canonical_id"],
                    "canonical_name": e["canonical_name"],
                    "entity_type": e["entity_type"],
                    "known_aliases": e.get("known_aliases", []),
                }
                for e in self.registry.entries
            ]

            for i in range(0, len(unresolved_strings), batch_size):
                batch = unresolved_strings[i : i + batch_size]
                logger.info(f"Querying Gemini API for batch of {len(batch)} unresolved strings ({i+1}-{i+len(batch)}/{len(unresolved_strings)})")
                
                try:
                    resolutions = self.gemini_client.normalize_batch(batch, registry_summary)
                except Exception as e:
                    logger.error(f"Failed batch normalization with Gemini API: {e}", exc_info=True)
                    raise e

                for raw_str, res in resolutions.items():
                    c_id = res.get("canonical_id")
                    if c_id and self.registry.find_by_id(c_id):
                        string_to_canonical_id[raw_str] = c_id
                    else:
                        # Register new proposed organization
                        c_name = res.get("canonical_name") or raw_str
                        e_type = res.get("entity_type") or "UNI"
                        new_entry = self.registry.register_organization(
                            canonical_name=c_name,
                            entity_type=e_type,
                            known_aliases=[raw_str],
                        )
                        string_to_canonical_id[raw_str] = new_entry["canonical_id"]

        # Step 3: Build normalized paper list with WITHIN-PAPER DEDUPLICATION
        normalized_papers: List[Dict[str, Any]] = []

        for p in raw_papers:
            paper_id = p.get("paper_id", "")
            title = p.get("title", "")
            raw_affs = p.get("raw_affiliations", [])

            canonical_ids_set: Set[str] = set()
            for aff in raw_affs:
                aff_clean = aff.strip() if aff else ""
                if aff_clean in string_to_canonical_id:
                    canonical_ids_set.add(string_to_canonical_id[aff_clean])

            normalized_papers.append({
                "paper_id": paper_id,
                "title": title,
                "canonical_ids": sorted(list(canonical_ids_set)),
            })

        norm_artifact = {
            "venue": venue_upper,
            "year": year,
            "total_papers": len(normalized_papers),
            "papers": normalized_papers,
        }

        try:
            with open(norm_path, "w", encoding="utf-8") as f:
                json.dump(norm_artifact, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved normalized dataset to {norm_path}")
            return norm_artifact
        except Exception as e:
            logger.error(f"Failed saving normalized dataset to {norm_path}", exc_info=True)
            raise e

