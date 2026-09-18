import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import NORMALIZED_DATA_DIR, OUTPUT_DATA_DIR
from src.registry.organization_registry import OrganizationRegistry
from src.utils import sanitize_venue_name

logger = logging.getLogger(__name__)


class MatrixExporter:
    """
    Aggregates normalized yearly paper records and exports a matrix CSV:
    Columns: canonical_id, canonical_name, entity_type, <start_year>, ..., <end_year>, total
    Enforces within-paper deduplication and paper-level multi-affiliation increments.
    Sorts descending by total count.
    """

    def __init__(
        self,
        registry: Optional[OrganizationRegistry] = None,
        normalized_dir: Path = NORMALIZED_DATA_DIR,
        output_dir: Path = OUTPUT_DATA_DIR,
    ):
        self.registry = registry or OrganizationRegistry()
        self.normalized_dir = normalized_dir
        self.output_dir = output_dir

    def export_matrix(
        self,
        venue: str,
        start_year: int,
        end_year: int,
        output_path: Optional[Path] = None,
    ) -> Path:
        venue_upper = venue.upper()
        clean_venue = sanitize_venue_name(venue)
        years = list(range(start_year, end_year + 1))

        if not output_path:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"{clean_venue}_affiliations_{start_year}_{end_year}.csv"

        logger.info(f"Aggregating matrix for {clean_venue} across years {start_year}..{end_year}")

        # Data structure: canonical_id -> year -> count
        counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for y in years:
            norm_file = self.normalized_dir / f"{clean_venue}_{y}_normalized.json"
            if not norm_file.exists():
                err_msg = f"Normalized data file for {clean_venue} {y} not found at {norm_file}. Run normalize subcommand first."
                logger.error(err_msg, exc_info=True)
                raise FileNotFoundError(err_msg)


            try:
                logger.info(f"Opening normalized data file for matrix reading: {norm_file.resolve()}")
                with open(norm_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Failed loading normalized file at {norm_file}", exc_info=True)
                raise e

            papers = data.get("papers", [])
            logger.info(f"Processing {len(papers)} normalized papers for {venue_upper} {y}")
            logger.info(f"Processing {len(papers)} normalized papers for {clean_venue} {y}")

            for p in papers:
                # canonical_ids is already deduplicated per paper by AffiliationNormalizer
                c_ids = set(p.get("canonical_ids", []))
                for cid in c_ids:
                    counts[cid][y] += 1

        # Build output rows
        rows: List[Dict[str, Any]] = []

        all_canonical_ids = set(counts.keys())
        for entry in self.registry.entries:
            all_canonical_ids.add(entry["canonical_id"])

        for cid in all_canonical_ids:
            reg_entry = self.registry.find_by_id(cid)
            if reg_entry:
                c_name = reg_entry["canonical_name"]
                e_type = reg_entry["entity_type"]
            else:
                c_name = cid
                e_type = "UNI"

            row: Dict[str, Any] = {
                "canonical_id": cid,
                "canonical_name": c_name,
                "entity_type": e_type,
            }

            row_total = 0
            for y in years:
                cnt = counts[cid][y]
                row[str(y)] = cnt
                row_total += cnt

            row["total"] = row_total

            # Include row if it has total > 0
            if row_total > 0:
                rows.append(row)

        # Sort descending by total, then by canonical_name
        rows.sort(key=lambda r: (-r["total"], r["canonical_name"]))

        # Fieldnames header
        fieldnames = ["canonical_id", "canonical_name", "entity_type"] + [str(y) for y in years] + ["total"]

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Opening CSV matrix file for writing: {output_path.resolve()}")
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            logger.info(f"Successfully exported affiliation matrix CSV to {output_path} ({len(rows)} institution records)")
            return output_path
        except Exception as e:
            logger.error(f"Failed exporting matrix CSV to {output_path}", exc_info=True)
            raise e

