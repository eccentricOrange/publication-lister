#!/usr/bin/env python3
import csv
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    CANONICAL_REGISTRY_PATH,
    CLEANED_OUTPUT_DATA_DIR,
    OPENALEX_SOURCES_CACHE_PATH,
    OUTPUT_DATA_DIR,
)

logger = logging.getLogger("build_site_data")


def parse_filename(filename: str) -> Tuple[str, int, int]:
    """Extracts venue, start_year, end_year from filename like ICRA_affiliations_2016_2026.csv."""
    m = re.match(r"^([A-Za-z0-9_\-]+)_affiliations_(\d{4})_(\d{4})\.csv$", filename)
    if m:
        return m.group(1).upper(), int(m.group(2)), int(m.group(3))
    return "", 0, 0


def build_site_data(
    cleaned_dir: Path = CLEANED_OUTPUT_DATA_DIR,
    output_dir: Path = OUTPUT_DATA_DIR,
    registry_path: Path = CANONICAL_REGISTRY_PATH,
    sources_cache_path: Path = OPENALEX_SOURCES_CACHE_PATH,
    target_json_path: Path = Path("visualisation/data/site_data.json"),
) -> Dict[str, Any]:
    """
    Scans matrix CSV files, merges split files per venue, cross-references canonical metadata and OpenAlex sources cache,
    and produces a consolidated JSON manifest for the visualisation web app.
    """
    logger.info("Building site data manifest for web visualization...")

    # 1. Locate CSV files (prefer cleaned_output, fallback to output)
    csv_files: List[Path] = []
    if cleaned_dir.exists():
        csv_files.extend(list(cleaned_dir.glob("*_affiliations_*.csv")))
    if not csv_files and output_dir.exists():
        csv_files.extend(list(output_dir.glob("*_affiliations_*.csv")))

    if not csv_files:
        logger.warning("No affiliation CSV files found to build visualization manifest.")

    # Group CSV files by venue
    venue_file_map: Dict[str, List[Path]] = {}
    for csv_file in csv_files:
        venue, start_yr, end_yr = parse_filename(csv_file.name)
        if venue:
            venue_file_map.setdefault(venue, []).append(csv_file)

    # 2. Load Canonical Registry metadata
    registry_map: Dict[str, Dict[str, Any]] = {}
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                reg_list = json.load(f)
                for entry in reg_list:
                    c_id = entry.get("canonical_id")
                    if c_id:
                        registry_map[c_id] = entry
        except Exception as e:
            logger.warning(f"Could not load canonical registry from {registry_path}: {e}")

    # 3. Load OpenAlex Sources Cache
    sources_cache: Dict[str, Any] = {}
    if sources_cache_path.exists():
        try:
            with open(sources_cache_path, "r", encoding="utf-8") as f:
                sources_cache = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load OpenAlex sources cache from {sources_cache_path}: {e}")

    # Helper to resolve source info from cache
    def get_source_info(v_name: str) -> Dict[str, Any]:
        info = {"source_ids": [], "doi_prefixes": [], "frequency": "annual", "url": None}
        v_upper = v_name.upper()

        raw_val = None
        for k, val in sources_cache.items():
            if str(k).upper() == v_upper or str(k).split()[0].upper() == v_upper:
                raw_val = val
                break

        if isinstance(raw_val, str):
            info["source_ids"] = [raw_val]
            info["url"] = f"https://openalex.org/sources/{raw_val}"
        elif isinstance(raw_val, dict):
            s_ids = raw_val.get("source_ids") or []
            if isinstance(s_ids, str):
                s_ids = [s_ids]
            info["source_ids"] = s_ids
            if s_ids:
                info["url"] = f"https://openalex.org/sources/{s_ids[0]}"
            info["doi_prefixes"] = raw_val.get("doi_prefixes") or []
            info["frequency"] = raw_val.get("frequency", "annual")

        return info

    venues_data: Dict[str, Any] = {}
    organisations_data: Dict[str, Any] = {}

    meta_headers = {"canonical_id", "canonical_name", "entity_type", "total"}

    for venue, files in sorted(venue_file_map.items()):
        logger.info(f"Processing venue '{venue}' across {len(files)} file(s)...")

        # Collect all years present across files for this venue
        all_years_set: Set[int] = set()
        file_rows_list: List[Tuple[List[str], List[Dict[str, str]]]] = []

        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                f_fields = reader.fieldnames or []
                f_rows = list(reader)
                for f_name in f_fields:
                    if f_name not in meta_headers and f_name.isdigit():
                        all_years_set.add(int(f_name))
                file_rows_list.append((f_fields, f_rows))

        sorted_years = [str(y) for y in sorted(list(all_years_set))]

        # Merge rows by canonical_id
        merged_entities: Dict[str, Dict[str, Any]] = {}

        for f_fields, f_rows in file_rows_list:
            for r in f_rows:
                c_id = (r.get("canonical_id") or "").strip()
                c_name = (r.get("canonical_name") or "").strip()
                e_type = (r.get("entity_type") or "UNI").strip()

                if not c_id and not c_name:
                    continue

                entity_key = c_id if c_id else c_name.lower()

                if entity_key not in merged_entities:
                    merged_entities[entity_key] = {
                        "canonical_id": c_id,
                        "canonical_name": c_name,
                        "entity_type": e_type,
                        "years": {y: 0 for y in sorted_years},
                    }

                # Copy non-zero yearly counts
                for y in sorted_years:
                    val_str = r.get(y, "0").strip()
                    if val_str.isdigit():
                        val = int(val_str)
                        if val > merged_entities[entity_key]["years"][y]:
                            merged_entities[entity_key]["years"][y] = val

        # Calculate totals and filter out 0 total entities
        final_venue_rows: List[Dict[str, Any]] = []
        venue_yearly_totals: Dict[str, int] = {y: 0 for y in sorted_years}

        for key, entity in merged_entities.items():
            tot = sum(entity["years"].values())
            entity["total"] = tot
            if tot > 0:
                final_venue_rows.append(entity)
                for y, cnt in entity["years"].items():
                    venue_yearly_totals[y] += cnt

        # Sort descending by total
        final_venue_rows.sort(key=lambda x: x["total"], reverse=True)

        src_info = get_source_info(venue)

        # Build Venue Entry
        venues_data[venue] = {
            "venue": venue,
            "years": sorted_years,
            "year_range": f"{sorted_years[0]} – {sorted_years[-1]}" if sorted_years else "",
            "total_publications": sum(venue_yearly_totals.values()),
            "yearly_totals": venue_yearly_totals,
            "total_institutions": len(final_venue_rows),
            "top_institution": {
                "canonical_name": final_venue_rows[0]["canonical_name"],
                "canonical_id": final_venue_rows[0]["canonical_id"],
                "total": final_venue_rows[0]["total"],
            } if final_venue_rows else None,
            "source_ids": src_info["source_ids"],
            "doi_prefixes": src_info["doi_prefixes"],
            "frequency": src_info["frequency"],
            "venue_url": src_info["url"],
            "rows": final_venue_rows,
        }

        # Populate Organisations global lookup map
        for row in final_venue_rows:
            cid = row["canonical_id"] or row["canonical_name"].lower()
            cname = row["canonical_name"]
            etype = row["entity_type"]

            if cid not in organisations_data:
                aliases = []
                if cid in registry_map:
                    aliases = registry_map[cid].get("known_aliases", [])

                organisations_data[cid] = {
                    "canonical_id": cid,
                    "canonical_name": cname,
                    "entity_type": etype,
                    "known_aliases": aliases,
                    "total_publications": 0,
                    "venues": {},
                    "openalex_source_ids": set(),
                    "doi_prefixes": set(),
                }

            org_entry = organisations_data[cid]
            org_entry["venues"][venue] = {
                "years": row["years"],
                "total": row["total"],
            }
            org_entry["total_publications"] += row["total"]
            if src_info["source_ids"]:
                org_entry["openalex_source_ids"].update(src_info["source_ids"])
            if src_info["doi_prefixes"]:
                org_entry["doi_prefixes"].update(src_info["doi_prefixes"])

    # Format sets to lists for JSON serialization in organisations_data
    final_orgs: Dict[str, Any] = {}
    for cid, org in sorted(organisations_data.items(), key=lambda x: x[1]["total_publications"], reverse=True):
        final_orgs[cid] = {
            "canonical_id": org["canonical_id"],
            "canonical_name": org["canonical_name"],
            "entity_type": org["entity_type"],
            "known_aliases": org["known_aliases"],
            "total_publications": org["total_publications"],
            "venues": org["venues"],
            "openalex_source_ids": sorted(list(org["openalex_source_ids"])),
            "doi_prefixes": sorted(list(org["doi_prefixes"])),
        }

    manifest = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "venues": venues_data,
        "organisations": final_orgs,
    }

    target_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    logger.info(
        f"Site data manifest generated successfully at {target_json_path.resolve()} "
        f"({len(venues_data)} venues, {len(final_orgs)} canonical organizations)."
    )
    return manifest


if __name__ == "__main__":
    from src.logger import setup_logging
    setup_logging()
    build_site_data()
