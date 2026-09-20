import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.cleaner.csv_cleaner import CSVCleaner
from src.config import DEFAULT_GEMINI_MODEL, NORMALIZED_DATA_DIR, RAW_DATA_DIR
from src.exporters.matrix_exporter import MatrixExporter
from src.extractors.ieee_xplore import IEEEExtractor
from src.extractors.openalex import OpenAlexExtractor
from src.extractors.scopus import ScopusExtractor
from src.normalizer.affiliation_normalizer import AffiliationNormalizer
from src.normalizer.gemini_client import GeminiClient
from src.registry.organization_registry import OrganizationRegistry
from src.runner.batch_config import BatchConfig

logger = logging.getLogger(__name__)


class BulkRunner:
    """
    Orchestrates high-performance multi-venue, multi-year pipeline:
    - Phase 1: Bulk dataset extraction across all configured venues & years.
    - Phase 2: Pooled global affiliation string normalization across ALL venues/years at once,
               maximizing cross-venue institution string overlap and minimizing Gemini LLM calls.
    - Phase 3: Aggregates and exports matrix CSVs to data/output/.
    - Phase 4: Cleans exported matrix CSVs with Gemini LLM into data/cleaned_output/.
    """

    def __init__(
        self,
        config: BatchConfig,
        registry: Optional[OrganizationRegistry] = None,
        gemini_client: Optional[GeminiClient] = None,
        model: Optional[str] = None,
        raw_dir: Path = RAW_DATA_DIR,
        normalized_dir: Path = NORMALIZED_DATA_DIR,
    ):
        self.config = config
        self.registry = registry or OrganizationRegistry()
        eff_model = model or config.model or DEFAULT_GEMINI_MODEL
        if gemini_client:
            self.gemini_client = gemini_client
            if eff_model:
                self.gemini_client.model = eff_model
        else:
            self.gemini_client = GeminiClient(model=eff_model)
        self.model = self.gemini_client.model
        self.normalizer = AffiliationNormalizer(
            registry=self.registry,
            gemini_client=self.gemini_client,
            raw_dir=raw_dir,
            normalized_dir=normalized_dir,
        )
        self.exporter = MatrixExporter(
            registry=self.registry,
            normalized_dir=normalized_dir,
        )
        self.raw_dir = raw_dir
        self.normalized_dir = normalized_dir

    def run_bulk_extraction(self, force: bool = False) -> List[Tuple[str, int]]:
        """Phase 1: Harvests raw paper datasets for all configured venues and years."""
        logger.info("=== Phase 1: Bulk Dataset Extraction ===")
        processed_pairs: List[Tuple[str, int]] = []

        for venue in self.config.venues:
            for year in self.config.years:
                if not self.config.is_year_active(venue, year):
                    logger.info(f"Skipping {venue} {year}: inactive year specified in batch configuration exceptions.")
                    extractor = OpenAlexExtractor(gemini_client=self.gemini_client)
                    extractor.append_raw_batch(venue, year, [], completed=True, next_cursor=None)
                    processed_pairs.append((venue, year))
                    continue

                overrides = self.config.get_overrides(venue, year)
                source = overrides.get("source", self.config.default_source)
                logger.info(f"Extracting raw data for {venue} {year} (source={source})...")

                if source == "openalex":
                    extractor = OpenAlexExtractor(gemini_client=self.gemini_client)
                    extractor.extract(
                        venue,
                        year,
                        force=force,
                        search_term=overrides.get("search_term"),
                        openalex_source_id=overrides.get("openalex_source_id"),
                        doi_prefix=overrides.get("doi_prefix"),
                    )
                elif source == "ieee":
                    extractor = IEEEExtractor()
                    extractor.extract(
                        venue,
                        year,
                        force=force,
                        query_term=overrides.get("query_term"),
                    )
                elif source == "scopus":
                    extractor = ScopusExtractor()
                    extractor.extract(venue, year, force=force)
                else:
                    err_msg = f"Unsupported extraction source '{source}' for {venue} {year}"
                    logger.error(err_msg)
                    raise ValueError(err_msg)

                processed_pairs.append((venue, year))

        logger.info(f"Phase 1 Complete: Extracted {len(processed_pairs)} venue/year datasets.")
        return processed_pairs

    def run_global_normalization(self, batch_size: int = 50, force: bool = False) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """
        Phase 2: Pools raw affiliation strings across ALL configured venues and years.
        Resolves strings globally in a single cumulative pass against OrganizationRegistry & Gemini LLM.
        """
        logger.info("=== Phase 2: Global Pooled Normalization ===")

        # 1. Collect all raw datasets and unique strings
        venue_year_raw_data: Dict[Tuple[str, int], Dict[str, Any]] = {}
        global_unique_strings: Set[str] = set()

        for venue in self.config.venues:
            for year in self.config.years:
                raw_file = self.raw_dir / venue / f"{venue}_{year}.json"
                if not raw_file.exists():
                    if not self.config.is_year_active(venue, year):
                        extractor = OpenAlexExtractor(gemini_client=self.gemini_client)
                        rdata = extractor.append_raw_batch(venue, year, [], completed=True, next_cursor=None)
                    else:
                        err_msg = f"Raw dataset file missing for {venue} {year} at {raw_file}. Run Phase 1 first."
                        logger.error(err_msg)
                        raise FileNotFoundError(err_msg)
                else:
                    with open(raw_file, "r", encoding="utf-8") as f:
                        rdata = json.load(f)

                venue_year_raw_data[(venue, year)] = rdata

                for p in rdata.get("papers", []):
                    for aff in p.get("raw_affiliations", []):
                        if aff and aff.strip():
                            global_unique_strings.add(aff.strip())

        logger.info(f"Collected {len(global_unique_strings)} total unique raw affiliation strings across {len(venue_year_raw_data)} venue/year datasets.")

        # 2. Local resolution first
        string_to_canonical_id: Dict[str, str] = {}
        unresolved_strings: List[str] = []

        strings_to_check = sorted(list(global_unique_strings))
        batch_matches = self.registry.batch_find_by_strings(strings_to_check)
        for raw_str in strings_to_check:
            match = batch_matches.get(raw_str)
            if match:
                string_to_canonical_id[raw_str] = match["canonical_id"]
            else:
                unresolved_strings.append(raw_str)

        logger.info(f"Local registry matched {len(string_to_canonical_id)} strings globally. {len(unresolved_strings)} unresolved strings remaining for LLM query.")

        # 3. Dynamic batching with inline compact context for remaining unresolved strings
        while unresolved_strings:
            still_unresolved = []
            newly_matched = 0
            strings_to_recheck = [raw_str for raw_str in unresolved_strings if raw_str not in string_to_canonical_id]
            recheck_matches = self.registry.batch_find_by_strings(strings_to_recheck)
            for raw_str in strings_to_recheck:
                match = recheck_matches.get(raw_str)
                if match:
                    string_to_canonical_id[raw_str] = match["canonical_id"]
                    newly_matched += 1
                else:
                    still_unresolved.append(raw_str)

            if newly_matched > 0:
                logger.info(f"Re-checking registry matched {newly_matched} additional strings locally. {len(still_unresolved)} remaining.")

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
                full_registry_summary,
                target_batch_size=batch_size,
            )

            logger.info(f"Querying Gemini API for pooled batch of {len(batch)} unresolved strings ({len(string_to_canonical_id)} resolved, {len(unresolved_strings)} remaining)")

            try:
                resolutions = self.gemini_client.normalize_batch(
                    batch,
                    canonical_registry_summary=registry_summary,
                )
            except Exception as e:
                logger.error(f"Failed pooled batch normalization with Gemini API: {e}", exc_info=True)
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
                else:
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

        # 4. Map global resolutions back to individual venue/year normalized files

        normalized_artifacts: Dict[Tuple[str, int], Dict[str, Any]] = {}

        for (venue, year), rdata in venue_year_raw_data.items():
            raw_papers = rdata.get("papers", [])
            norm_path = self.normalizer.get_normalized_file_path(venue, year)

            norm_mappings: Dict[str, str] = {}
            norm_papers: List[Dict[str, Any]] = []

            for p in raw_papers:
                paper_id = p.get("paper_id", "")
                title = p.get("title", "")
                raw_affs = p.get("raw_affiliations", [])

                canonical_ids_set: Set[str] = set()
                for aff in raw_affs:
                    aff_clean = aff.strip() if aff else ""
                    if aff_clean in string_to_canonical_id:
                        c_id = string_to_canonical_id[aff_clean]
                        canonical_ids_set.add(c_id)
                        norm_mappings[aff_clean] = c_id

                norm_papers.append({
                    "paper_id": paper_id,
                    "title": title,
                    "canonical_ids": sorted(list(canonical_ids_set)),
                })

            artifact = {
                "venue": venue,
                "year": year,
                "completed": True,
                "total_papers": len(norm_papers),
                "resolved_mappings": norm_mappings,
                "papers": norm_papers,
            }

            with open(norm_path, "w", encoding="utf-8") as f:
                json.dump(artifact, f, indent=2, ensure_ascii=False)

            # Update step status in raw file
            steps = rdata.get("steps", {})
            steps["extraction"] = {"completed": True, "total_papers": len(raw_papers)}
            steps["normalization"] = {"completed": True, "normalized_file": str(norm_path.resolve())}
            rdata["steps"] = steps

            raw_file = self.raw_dir / venue / f"{venue}_{year}.json"
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump(rdata, f, indent=2, ensure_ascii=False)

            normalized_artifacts[(venue, year)] = artifact
            logger.info(f"Saved normalized dataset for {venue} {year} to {norm_path} ({len(norm_papers)} papers)")

        logger.info("Phase 2 Complete: Global pooled normalization finished for all venue datasets.")
        return normalized_artifacts

    def run_bulk_export(self) -> List[Path]:
        """Phase 3: Exports matrix CSVs for all configured venues across years."""
        logger.info("=== Phase 3: Bulk Matrix Export ===")
        exported_files: List[Path] = []

        start_year = min(self.config.years)
        end_year = max(self.config.years)

        for venue in self.config.venues:
            csv_path = self.exporter.export_matrix(venue, start_year, end_year)
            exported_files.append(csv_path)
            logger.info(f"Exported matrix for venue '{venue}' ({start_year}..{end_year}): {csv_path}")

        logger.info(f"Phase 3 Complete: Exported {len(exported_files)} CSV matrix files.")
        return exported_files

    def run_bulk_clean(self, exported_files: Optional[List[Path]] = None) -> List[Path]:
        """Phase 4: Cleans exported matrix CSVs with Gemini LLM into data/cleaned_output/."""
        logger.info("=== Phase 4: Bulk Matrix Cleaning ===")
        cleaner = CSVCleaner(registry=self.registry, gemini_client=self.gemini_client)
        cleaned_files = cleaner.clean_all(input_dir=self.exporter.output_dir)
        logger.info(f"Phase 4 Complete: Cleaned {len(cleaned_files)} CSV matrix files into {cleaner.output_dir}.")
        return cleaned_files

    def run_all(self, force: bool = False) -> List[Path]:
        """Executes full 4-phase bulk pipeline: Extraction -> Global Normalization -> Export -> Cleaning."""
        logger.info(f"Starting Bulk Pipeline across {len(self.config.venues)} venues and years {min(self.config.years)}..{max(self.config.years)}")
        self.run_bulk_extraction(force=force)
        self.run_global_normalization(force=force)
        exported = self.run_bulk_export()
        cleaned = self.run_bulk_clean(exported)
        logger.info("Bulk Pipeline execution finished successfully.")
        return cleaned

