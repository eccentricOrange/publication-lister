import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

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
    # Remove special characters
    clean_name = re.sub(r"[^\w\s]", "", name).upper()
    words = clean_name.split()

    if not words:
        return "XXXXXX"

    if len(words) == 1:
        slug = words[0][:6]
    else:
        # Pick capital initials or word prefixes
        letters = [w[0] for w in words if w]
        if len(letters) >= 6:
            slug = "".join(letters[:6])
        else:
            slug = "".join(letters)
            for w in words:
                if len(slug) < 6 and len(w) > 1:
                    slug += w[1 : 6 - len(slug) + 1]
                if len(slug) >= 6:
                    break

    slug = re.sub(r"[^A-Z0-9]", "", slug).upper()
    slug = slug.ljust(6, "X")[:6]
    return slug


class OrganizationRegistry:
    """
    Manages persistent central registry of canonical organization entities.
    Enforces strict ID schema: [TYPE:3]-[ID:5]-[SLUG:6]
    """

    def __init__(self, registry_path: Path = CANONICAL_REGISTRY_PATH):
        self.registry_path = registry_path
        self.entries: List[Dict[str, Any]] = []
        self._lookup_map: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Loads canonical organization records from JSON file."""
        if not self.registry_path.exists():
            logger.warning(f"Registry file not found at {self.registry_path}. Initializing empty registry.")
            self.entries = []
            self._rebuild_lookup_map()
            return

        try:
            logger.info(f"Opening canonical organization registry file for reading: {self.registry_path.resolve()}")
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
            self._rebuild_lookup_map()
            logger.info(f"Loaded {len(self.entries)} canonical organization entries from {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to load canonical organization registry from {self.registry_path}", exc_info=True)
            raise e

    def save(self) -> None:
        """Saves current registry entries to JSON file."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Opening canonical organization registry file for writing: {self.registry_path.resolve()}")
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.entries)} canonical entries to {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to save canonical registry to {self.registry_path}", exc_info=True)
            raise e

    def _normalize_key(self, text: str) -> str:
        """Normalize string for lookup matching."""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _rebuild_lookup_map(self) -> None:
        """Rebuilds fast lookup map for alias/name matching."""
        self._lookup_map = {}
        for entry in self.entries:
            c_id = entry["canonical_id"]
            c_name = entry["canonical_name"]
            self._lookup_map[c_id.upper()] = entry
            self._lookup_map[self._normalize_key(c_name)] = entry
            for alias in entry.get("known_aliases", []):
                self._lookup_map[self._normalize_key(alias)] = entry

    def find_by_string(self, raw_string: str) -> Optional[Dict[str, Any]]:
        """
        Looks up a raw affiliation string in canonical registry.
        Checks canonical_id first, then canonical name and known aliases via exact & longest substring match.
        """
        if not raw_string:
            return None

        clean_str = raw_string.strip()
        if clean_str.upper() in self._lookup_map:
            return self._lookup_map[clean_str.upper()]

        norm_key = self._normalize_key(clean_str)
        if norm_key in self._lookup_map:
            return self._lookup_map[norm_key]

        # Stopwords to ignore for standalone substring matching
        generic_stopwords = {"university", "college", "institute", "school", "department", "center", "centre", "laboratory", "lab", "inc", "ltd", "corp", "corporation", "llc", "group", "faculty", "academy"}

        # Try matching known canonical names & aliases as substrings inside norm_key
        best_match = None
        longest_match_len = 0

        for entry in self.entries:
            for alias in [entry["canonical_name"]] + entry.get("known_aliases", []):
                alias_norm = self._normalize_key(alias)
                if not alias_norm or alias_norm in generic_stopwords:
                    continue
                # Require alias length >= 4 to avoid short acronym false positives
                if len(alias_norm) >= 4 and alias_norm in norm_key:
                    if len(alias_norm) > longest_match_len:
                        best_match = entry
                        longest_match_len = len(alias_norm)

        return best_match

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

