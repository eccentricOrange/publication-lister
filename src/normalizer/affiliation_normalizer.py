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
    Integrates Gemini Server-Side Context Caching (cachedContents) and local double-checking.
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
        batch_size: int = 50,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Normalizes raw paper affiliation dataset for a venue and year.
        Dumps output to data/normalized/<venue>_<year>_normalized.json.
        """
        venue_upper = venue.upper()
        norm_path = self.get_normalized_file_path(venue_upper, year)

        # Check if fully completed normalized dataset exists
        cached_resolved_mappings: Dict[str, str] = {}
        if not force and norm_path.exists() and norm_path.stat().st_size > 0:
            try:
                logger.info(f"Opening normalized file for reading cached data: {norm_path.resolve()}")
                with open(norm_path, "r", encoding="utf-8") as f:
                    norm_data = json.load(f)
                if norm_data.get("completed", False):
                    logger.info(f"Normalized dataset for {venue_upper} {year} is fully completed at {norm_path}")
                    return norm_data
                cached_resolved_mappings = norm_data.get("resolved_mappings", {})
                logger.info(f"Resuming partial normalization for {venue_upper} {year} ({len(cached_resolved_mappings)} strings previously resolved)")
            except Exception as e:
                logger.warning(f"Could not read existing normalized dataset at {norm_path}: {e}")

        # Load raw file
        raw_file = self.raw_dir / venue_upper / f"{venue_upper}_{year}.json"
        if not raw_file.exists():
            err_msg = f"Raw dataset not found at {raw_file}. Run extract subcommand first."
            logger.error(err_msg, exc_info=True)
            raise FileNotFoundError(err_msg)

        try:
            logger.info(f"Opening raw dataset file for reading: {raw_file.resolve()}")
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

        # Step 1: Resolve locally using existing registry and cached mappings
        string_to_canonical_id: Dict[str, str] = dict(cached_resolved_mappings)
        unresolved_strings: List[str] = []

        for raw_str in sorted(list(unique_raw_strings)):
            if raw_str in string_to_canonical_id:
                continue
            match = self.registry.find_by_string(raw_str)
            if match:
                string_to_canonical_id[raw_str] = match["canonical_id"]
            else:
                unresolved_strings.append(raw_str)

        logger.info(f"Local registry & cache matched {len(string_to_canonical_id)} strings. {len(unresolved_strings)} unresolved strings remaining.")

        # Helper to save normalized checkpoint
        def _save_checkpoint(completed: bool = False, normalized_papers_list: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
            artifact = {
                "venue": venue_upper,
                "year": year,
                "completed": completed,
                "total_papers": len(normalized_papers_list) if normalized_papers_list is not None else len(raw_papers),
                "resolved_mappings": string_to_canonical_id,
                "papers": normalized_papers_list or [],
            }
            try:
                logger.info(f"Opening normalized dataset file for writing: {norm_path.resolve()}")
                with open(norm_path, "w", encoding="utf-8") as f:
                    json.dump(artifact, f, indent=2, ensure_ascii=False)
            except Exception as exc:
                logger.warning(f"Failed saving normalization checkpoint to {norm_path}: {exc}")
            
            # Also update raw paper file with step status
            try:
                steps = raw_data.get("steps", {})
                steps["extraction"] = {"completed": raw_data.get("completed", True), "total_papers": len(raw_papers)}
                steps["normalization"] = {"completed": completed, "normalized_file": str(norm_path.resolve())}
                raw_data["steps"] = steps
                logger.info(f"Opening raw dataset file for updating step status: {raw_file.resolve()}")
                with open(raw_file, "w", encoding="utf-8") as f:
                    json.dump(raw_data, f, indent=2, ensure_ascii=False)
            except Exception as exc:
                logger.warning(f"Failed updating step status in raw file {raw_file}: {exc}")
            
            return artifact

        # Initialize server-side context cache if unresolved strings exist
        cached_content_id: Optional[str] = None
        if unresolved_strings and self.gemini_client.api_key:
            try:
                full_registry_summary = [
                    {
                        "canonical_id": e["canonical_id"],
                        "canonical_name": e["canonical_name"],
                        "entity_type": e["entity_type"],
                        "known_aliases": e.get("known_aliases", []),
                    }
                    for e in self.registry.entries
                ]
                cached_content_id = self.gemini_client.create_cached_context(full_registry_summary)
            except Exception as exc:
                logger.warning(f"Failed creating Gemini server-side context cache: {exc}. Proceeding with inline prompt context.")

        new_orgs_since_cache_sync = 0

        # Step 2: Batch unresolved strings and query Gemini API
        while unresolved_strings:
            # Re-check remaining unresolved strings against updated local registry
            still_unresolved = []
            newly_matched_count = 0
            for raw_str in unresolved_strings:
                if raw_str in string_to_canonical_id:
                    continue
                match = self.registry.find_by_string(raw_str)
                if match:
                    string_to_canonical_id[raw_str] = match["canonical_id"]
                    newly_matched_count += 1
                else:
                    still_unresolved.append(raw_str)
            
            if newly_matched_count > 0:
                logger.info(f"Re-checking registry matched {newly_matched_count} additional strings locally. {len(still_unresolved)} remaining.")

            unresolved_strings = still_unresolved
            if not unresolved_strings:
                break

            full_registry_summary = [
                {
                    "canonical_id": e["canonical_id"],
                    "canonical_name": e["canonical_name"],
                    "entity_type": e["entity_type"],
                    "known_aliases": e.get("known_aliases", []),
                }
                for e in self.registry.entries
            ]

            batch, registry_summary = self.gemini_client.calculate_dynamic_batch(
                unresolved_strings,
                full_registry_summary if not cached_content_id else None,
                target_batch_size=batch_size if cached_content_id else 50,
            )

            logger.info(f"Querying Gemini API for batch of {len(batch)} unresolved strings (Cached Context: {cached_content_id or 'None'}, {len(string_to_canonical_id)} resolved locally, {len(unresolved_strings)} remaining)")

            try:
                resolutions = self.gemini_client.normalize_batch(
                    batch,
                    cached_content=cached_content_id,
                    canonical_registry_summary=registry_summary if not cached_content_id else None,
                )
            except Exception as e:
                logger.error(f"Failed batch normalization with Gemini API: {e}", exc_info=True)
                _save_checkpoint(completed=False)
                raise e

            for raw_str in batch:
                res = resolutions.get(raw_str) or resolutions.get(raw_str.strip())
                if not res:
                    norm_k = raw_str.strip().lower()
                    for k, v in resolutions.items():
                        if str(k).strip().lower() == norm_k:
                            res = v
                            break

                if res and isinstance(res, dict):
                    c_id = res.get("canonical_id")
                    if c_id and self.registry.find_by_id(c_id):
                        string_to_canonical_id[raw_str] = c_id
                        continue

                    c_name = (res.get("canonical_name") or raw_str).strip()
                    
                    # Double-check local registry before registering new entity (prevents duplicates across cached calls)
                    match = self.registry.find_by_string(c_name) or self.registry.find_by_string(raw_str.strip())
                    if match:
                        string_to_canonical_id[raw_str] = match["canonical_id"]
                        continue

                    e_type = res.get("entity_type") or "UNI"
                    new_entry = self.registry.register_organization(
                        canonical_name=c_name,
                        entity_type=e_type,
                        known_aliases=[raw_str.strip()],
                    )
                    string_to_canonical_id[raw_str] = new_entry["canonical_id"]
                    new_orgs_since_cache_sync += 1
                else:
                    # Fallback if LLM omitted key: double check local registry or register raw string
                    match = self.registry.find_by_string(raw_str.strip())
                    if match:
                        string_to_canonical_id[raw_str] = match["canonical_id"]
                    else:
                        new_entry = self.registry.register_organization(
                            canonical_name=raw_str.strip(),
                            entity_type="UNI",
                            known_aliases=[raw_str.strip()],
                        )
                        string_to_canonical_id[raw_str] = new_entry["canonical_id"]
                        new_orgs_since_cache_sync += 1
            
            # Re-sync server-side context cache if local registry has grown significantly (>50 new entities)
            if cached_content_id and new_orgs_since_cache_sync >= 50:
                try:
                    updated_registry_summary = [
                        {
                            "canonical_id": e["canonical_id"],
                            "canonical_name": e["canonical_name"],
                            "entity_type": e["entity_type"],
                            "known_aliases": e.get("known_aliases", []),
                        }
                        for e in self.registry.entries
                    ]
                    cached_content_id = self.gemini_client.create_cached_context(updated_registry_summary)
                    new_orgs_since_cache_sync = 0
                except Exception as exc:
                    logger.warning(f"Could not re-sync Gemini server-side context cache: {exc}")

            # Checkpoint after each batch
            _save_checkpoint(completed=False)

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

        final_artifact = _save_checkpoint(completed=True, normalized_papers_list=normalized_papers)
        logger.info(f"Successfully completed and saved normalized dataset to {norm_path}")
        return final_artifact
