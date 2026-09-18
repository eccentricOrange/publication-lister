import os
import concurrent.futures
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from src.config import CANONICAL_REGISTRY_PATH

logger = logging.getLogger(__name__)

VALID_ENTITY_TYPES = {"UNI", "COM", "LAB", "GOV"}


def generate_slug(name: str) -> str:
    """
    Generates a 6-character uppercase alphanumeric deterministic short code / slug from organization name.
    e.g. 'University of California, San Diego' -> 'UCSDCA'
         'Google LLC' -> 'GOOGUS'
         'NASA Jet Propulsion Laboratory' -> 'NASAJPL'
    """
    clean = re.sub(r"[^a-zA-Z0-9]", "", name).upper()
    if len(clean) >= 6:
        return clean[:6]
    return clean.ljust(6, "X")



class OrganizationRegistry:
    """
    Central Manager for Canonical Organizations Registry (data/canonical_organizations.json).
    Ensures unique canonical IDs ([TYPE:3]-[ID:5]-[SLUG:6]), alias deduplication,
    pre-indexed fast substring matching, and parallel multithreaded batch processing.
    """

    def __init__(self, registry_path: Path = CANONICAL_REGISTRY_PATH):
        self.registry_path = registry_path
        self.entries: List[Dict[str, Any]] = []
        self._lookup_map: Dict[str, Dict[str, Any]] = {}
        self._searchable_aliases: List[Tuple[str, Dict[str, Any]]] = []
        self.load_registry()

    def load_registry(self) -> None:
        """Loads canonical organization registry from JSON file."""
        if not self.registry_path.exists() or self.registry_path.stat().st_size == 0:
            logger.info(f"Canonical organization registry file not found at {self.registry_path}. Initializing empty registry.")
            self.entries = []
            self._rebuild_lookup_map()
            return

        try:
            logger.info(f"Opening file for reading canonical registry: {self.registry_path.resolve()}")
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
            self._rebuild_lookup_map()
            logger.info(f"Successfully loaded {len(self.entries)} canonical organization entities from {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to load canonical registry file at {self.registry_path}", exc_info=True)
            raise e

    def save_registry(self) -> None:
        """Saves canonical organization registry to JSON file."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Opening canonical organization registry file for writing: {self.registry_path.resolve()}")
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.entries)} canonical entries to {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to save canonical registry to {self.registry_path}", exc_info=True)
            raise e

    def save(self) -> None:
        """Alias for save_registry."""
        self.save_registry()


    def _normalize_key(self, text: str) -> str:
        """Normalize string for lookup matching."""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _rebuild_lookup_map(self) -> None:
        """Rebuilds fast lookup map and pre-indexes searchable aliases sorted descending by length."""
        self._lookup_map = {}
        generic_stopwords = {
            "university", "college", "institute", "school", "department", "center", "centre", 
            "laboratory", "lab", "inc", "ltd", "corp", "corporation", "llc", "group", "faculty", "academy"
        }
        
        searchable_list = []
        for entry in self.entries:
            c_id = entry["canonical_id"]
            c_name = entry["canonical_name"]
            self._lookup_map[c_id.upper()] = entry
            
            c_name_norm = self._normalize_key(c_name)
            self._lookup_map[c_name_norm] = entry

            seen_norms = set()
            for alias in [c_name] + entry.get("known_aliases", []):
                alias_norm = self._normalize_key(alias)
                self._lookup_map[alias_norm] = entry
                if alias_norm and alias_norm not in generic_stopwords and len(alias_norm) >= 4:
                    if alias_norm not in seen_norms:
                        seen_norms.add(alias_norm)
                        searchable_list.append((alias_norm, entry))

        # Sort aliases descending by length so the first match in find_by_string is guaranteed to be the longest match
        searchable_list.sort(key=lambda x: len(x[0]), reverse=True)
        self._searchable_aliases = searchable_list

    def find_by_string(self, raw_string: str) -> Optional[Dict[str, Any]]:
        """
        Looks up a raw affiliation string in canonical registry.
        Checks canonical_id first, then exact canonical name & aliases, and finally pre-indexed longest substring match.
        """
        if not raw_string:
            return None

        clean_str = raw_string.strip()
        if clean_str.upper() in self._lookup_map:
            return self._lookup_map[clean_str.upper()]

        norm_key = self._normalize_key(clean_str)
        if norm_key in self._lookup_map:
            return self._lookup_map[norm_key]

        len_norm = len(norm_key)
        for alias_norm, entry in self._searchable_aliases:
            if len(alias_norm) <= len_norm and alias_norm in norm_key:
                # Guaranteed to be the longest match because _searchable_aliases is sorted descending by length
                return entry

        return None

    def batch_find_by_strings(
        self,
        raw_strings: List[str],
        max_workers: Optional[int] = None,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Parallelized lookup for a list of raw affiliation strings against canonical registry.
        Uses ThreadPoolExecutor to process strings concurrently across CPU cores.
        """
        if not raw_strings:
            return {}

        if len(raw_strings) < 20:
            return {s: self.find_by_string(s) for s in raw_strings}

        workers = max_workers or min(32, (os.cpu_count() or 4) * 4)
        results: Dict[str, Optional[Dict[str, Any]]] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_string = {executor.submit(self.find_by_string, s): s for s in raw_strings}
            for future in concurrent.futures.as_completed(future_to_string):
                s = future_to_string[future]
                results[s] = future.result()

        return results

    def find_by_id(self, canonical_id: str) -> Optional[Dict[str, Any]]:
        """Finds entry by exact canonical ID."""
        for entry in self.entries:
            if entry["canonical_id"].upper() == canonical_id.strip().upper():
                return entry
        return None

    def generate_next_id(self, entity_type: str, canonical_name: str) -> str:
        """
        Generates next canonical ID following [TYPE:3]-[ID:5]-[SLUG:6].
        """
        entity_type = entity_type.upper()
        if entity_type not in VALID_ENTITY_TYPES:
            raise ValueError(f"Invalid entity type '{entity_type}'. Must be one of {VALID_ENTITY_TYPES}")

        # Find max integer ID across all existing entries
        max_id = 0
        for entry in self.entries:
            cid = entry.get("canonical_id", "")
            parts = cid.split("-")
            if len(parts) >= 2 and parts[1].isdigit():
                max_id = max(max_id, int(parts[1]))

        next_seq = max_id + 1
        id_str = f"{next_seq:05d}"
        slug_str = generate_slug(canonical_name)
        return f"{entity_type}-{id_str}-{slug_str}"

    def register_organization(
        self,
        canonical_name: str,
        entity_type: str,
        known_aliases: Optional[List[str]] = None,
        canonical_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Registers a new canonical organization or updates existing one with new aliases.
        """
        entity_type = entity_type.upper()
        if entity_type not in VALID_ENTITY_TYPES:
            raise ValueError(f"Invalid entity type '{entity_type}'. Must be one of {VALID_ENTITY_TYPES}")

        existing = self.find_by_string(canonical_name)
        if existing:
            # Update known aliases
            aliases = set(existing.get("known_aliases", []))
            aliases.add(canonical_name)
            if known_aliases:
                aliases.update(known_aliases)
            existing["known_aliases"] = sorted(list(aliases))
            self._rebuild_lookup_map()
            self.save()
            return existing

        if not canonical_id:
            canonical_id = self.generate_next_id(entity_type, canonical_name)

        aliases = set(known_aliases or [])
        aliases.add(canonical_name)

        new_entry = {
            "canonical_id": canonical_id,
            "canonical_name": canonical_name,
            "entity_type": entity_type,
            "known_aliases": sorted(list(aliases)),
        }

        self.entries.append(new_entry)
        self._rebuild_lookup_map()
        self.save()
        logger.info(f"Registered new canonical entity: {canonical_id} - {canonical_name} ({entity_type})")
        return new_entry

