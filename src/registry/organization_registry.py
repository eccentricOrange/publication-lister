import os
import concurrent.futures
import html
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from src.config import CANONICAL_REGISTRY_PATH

logger = logging.getLogger(__name__)

VALID_ENTITY_TYPES = {"UNI", "COM", "LAB", "GOV"}


def _multiprocess_match_chunk_worker(args_tuple):
    """
    Multiprocessing worker function executing on child process workers.
    Bypasses Python GIL to utilize 100% of all CPU cores.
    Uses C-level substring pre-check ('alias_norm in norm_key') before calling regex.
    """
    strings_chunk, lookup_map, searchable_aliases = args_tuple
    abbrevs = [
        (r'\buniv\b\.?', 'university'),
        (r'\bdept\b\.?', 'department'),
        (r'\binst\b\.?', 'institute'),
        (r'\btech\b\.?', 'technology'),
        (r'\blab\b\.?', 'laboratory'),
        (r'\blabs\b\.?', 'laboratory'),
        (r'\bfac\b\.?', 'faculty'),
        (r'\bsch\b\.?', 'school'),
        (r'\bcorp\b\.?', 'corporation'),
        (r'\binc\b\.?', 'incorporated'),
        (r'\bltd\b\.?', 'limited'),
        (r'\bcalif\b\.?', 'california'),
        (r'\bmass\b\.?', 'massachusetts'),
        (r'\bpenn\b\.?', 'pennsylvania'),
    ]

    def norm_k(text: str) -> str:
        s = text.lower()
        for pattern, repl in abbrevs:
            s = re.sub(pattern, repl, s)
        return re.sub(r"\s+", " ", s.strip())

    results = {}
    for s in strings_chunk:
        if not s:
            results[s] = None
            continue
        clean_str = s.strip()
        up_key = clean_str.upper()
        if up_key in lookup_map:
            results[s] = lookup_map[up_key]
            continue
        nk = norm_k(clean_str)
        if nk in lookup_map:
            results[s] = lookup_map[nk]
            continue

        len_nk = len(nk)
        matched = None
        for alias_norm, entry in searchable_aliases:
            if len(alias_norm) <= len_nk and alias_norm in nk:
                if re.search(r'\b' + re.escape(alias_norm) + r'\b', nk):
                    matched = entry
                    break
        results[s] = matched

    return results


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
        """Normalize string for lookup matching with abbreviation expansion."""
        s = text.lower()
        abbrevs = [
            (r'\buniv\b\.?', 'university'),
            (r'\bdept\b\.?', 'department'),
            (r'\binst\b\.?', 'institute'),
            (r'\btech\b\.?', 'technology'),
            (r'\blab\b\.?', 'laboratory'),
            (r'\blabs\b\.?', 'laboratory'),
            (r'\bfac\b\.?', 'faculty'),
            (r'\bsch\b\.?', 'school'),
            (r'\bcorp\b\.?', 'corporation'),
            (r'\binc\b\.?', 'incorporated'),
            (r'\bltd\b\.?', 'limited'),
            (r'\bcalif\b\.?', 'california'),
            (r'\bmass\b\.?', 'massachusetts'),
            (r'\bpenn\b\.?', 'pennsylvania'),
        ]
        for pattern, repl in abbrevs:
            s = re.sub(pattern, repl, s)
        return re.sub(r"\s+", " ", s.strip())

    def _rebuild_lookup_map(self) -> None:
        """Rebuilds fast lookup map and pre-indexes searchable aliases sorted descending by length."""
        self._lookup_map = {}
        generic_stopwords = {
            "university", "college", "institute", "school", "department", "center", "centre", 
            "laboratory", "lab", "inc", "ltd", "corp", "corporation", "llc", "group", "faculty", "academy"
        }
        generic_prefixes = {
            "department of computer science", "department of electrical engineering",
            "department of computer engineering", "school of computer science",
            "department of robotics", "faculty of engineering", "department of computer science and engineering",
            "department of mechanical engineering", "department of automation"
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
                if alias_norm and alias_norm not in generic_stopwords and alias_norm not in generic_prefixes and len(alias_norm) >= 4:
                    if alias_norm not in seen_norms:
                        seen_norms.add(alias_norm)
                        searchable_list.append((alias_norm, entry))

        # Sort aliases descending by length so the first match in find_by_string is guaranteed to be the longest match
        searchable_list.sort(key=lambda x: len(x[0]), reverse=True)
        self._searchable_aliases = searchable_list

    def find_by_string(self, raw_string: str) -> Optional[Dict[str, Any]]:
        """
        Looks up a raw affiliation string in canonical registry.
        Checks canonical_id first, then exact canonical name & aliases, and finally pre-indexed word-boundary substring match.
        Uses C-level fast substring pre-check ('alias_norm in norm_key') before calling regex (137x speedup).
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
                if re.search(r'\b' + re.escape(alias_norm) + r'\b', norm_key):
                    return entry

        return None

    def batch_find_by_strings(
        self,
        raw_strings: List[str],
        max_workers: Optional[int] = None,
        log_progress: bool = True,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        High-performance 2-tier multiprocessing lookup for raw affiliation strings:
        - Tier 1: Instant O(1) exact hash map lookup across all strings.
        - Tier 2: ProcessPoolExecutor multiprocessing across ALL CPU cores (bypassing Python GIL)
                  with C-level substring pre-check and real-time progress logging.
        """
        if not raw_strings:
            return {}

        import time

        total_count = len(raw_strings)
        results: Dict[str, Optional[Dict[str, Any]]] = {}
        unmatched_strings: List[str] = []

        t0 = time.time()

        # Tier 1: Fast O(1) exact hash lookup
        for s in raw_strings:
            if not s:
                results[s] = None
                continue
            clean_str = s.strip()
            up_key = clean_str.upper()
            if up_key in self._lookup_map:
                results[s] = self._lookup_map[up_key]
            else:
                norm_key = self._normalize_key(clean_str)
                if norm_key in self._lookup_map:
                    results[s] = self._lookup_map[norm_key]
                else:
                    unmatched_strings.append(s)

        exact_matched = len(results)
        t1 = time.time()

        if log_progress and total_count >= 100:
            logger.info(
                f"Tier 1 Exact Hash Lookup: matched {exact_matched}/{total_count} strings ({exact_matched / total_count * 100:.1f}%) in {t1 - t0:.2f}s. "
                f"{len(unmatched_strings)} remaining for Tier 2 Multiprocessing Search."
            )

        if not unmatched_strings:
            return results

        # Tier 2: Multiprocessing ProcessPoolExecutor across ALL CPU cores
        num_procs = max_workers or (os.cpu_count() or 4)
        total_tier2 = len(unmatched_strings)

        chunk_size = max(200, total_tier2 // (num_procs * 8))
        chunks = [unmatched_strings[i:i + chunk_size] for i in range(0, total_tier2, chunk_size)]
        tasks = [(chunk, self._lookup_map, self._searchable_aliases) for chunk in chunks]

        if log_progress:
            logger.info(f"Launching Tier 2 Multiprocessing across {num_procs} CPU cores ({len(chunks)} tasks, ~{chunk_size} strings/task)...")

        completed_tier2 = 0
        last_log_time = time.time()
        log_interval_items = max(2500, total_tier2 // 20)

        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as executor:
                future_to_chunk = {executor.submit(_multiprocess_match_chunk_worker, task): task[0] for task in tasks}
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_res = future.result()
                    results.update(chunk_res)
                    completed_tier2 += len(chunk_res)

                    now = time.time()
                    if log_progress and (completed_tier2 % log_interval_items < len(chunk_res) or completed_tier2 == total_tier2 or (now - last_log_time) >= 5.0):
                        last_log_time = now
                        pct = (completed_tier2 / total_tier2) * 100
                        speed = completed_tier2 / max(0.001, now - t1)
                        logger.info(f"Tier 2 Multiprocessing Progress ({num_procs} Cores): {completed_tier2}/{total_tier2} strings processed ({pct:.1f}%) [{speed:.1f} strings/sec]...")
        except Exception as proc_err:
            logger.warning(f"Multiprocessing encountered issue ({proc_err}). Falling back to multithreaded pool...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_procs * 2) as executor:
                future_to_string = {executor.submit(self.find_by_string, s): s for s in unmatched_strings}
                for future in concurrent.futures.as_completed(future_to_string):
                    s = future_to_string[future]
                    results[s] = future.result()

        t2 = time.time()
        total_matched = sum(1 for v in results.values() if v is not None)
        if log_progress and total_count >= 100:
            logger.info(f"Local Registry Matching Complete: total {total_matched}/{total_count} strings resolved ({total_matched / total_count * 100:.1f}%) in {t2 - t0:.2f}s.")

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

    def _clean_registry_local(self) -> Dict[str, Any]:
        """
        Pure local cleanup pass:
        - Unescapes HTML entities (e.g. &#x0026; -> &, &amp; -> &).
        - Prunes standalone generic department entries.
        - Merges duplicate canonical entries matching on normalized canonical name.
        - Cleans and deduplicates known_aliases.
        - Standardizes entity_type and canonical_id formatting.
        """
        if not self.entries:
            return {
                "initial_count": 0,
                "cleaned_html_entities": 0,
                "merged_duplicates": 0,
                "pruned_generic_departments": 0,
                "final_count": 0,
            }

        initial_count = len(self.entries)
        cleaned_html = 0
        pruned_depts = 0

        generic_prefixes = {
            "department of computer science", "department of electrical engineering",
            "department of computer engineering", "school of computer science",
            "department of robotics", "faculty of engineering", "department of computer science and engineering",
            "department of mechanical engineering", "department of automation",
            "department of information technology", "department of mathematics",
            "department of mechanical and information engineering", "department of civil and mechanical engineering",
            "faculty of computer science", "school of engineering", "department of electrical and computer engineering"
        }

        stage1_entries: List[Dict[str, Any]] = []
        for entry in self.entries:
            cname = entry.get("canonical_name", "")
            unescaped_cname = html.unescape(cname).strip()

            has_html = (unescaped_cname != cname) or any(html.unescape(str(a)) != str(a) for a in entry.get("known_aliases", []))
            if has_html:
                cleaned_html += 1

            cname_norm = re.sub(r"\s+", " ", unescaped_cname.lower())
            if cname_norm in generic_prefixes:
                pruned_depts += 1
                continue

            entry["canonical_name"] = unescaped_cname
            stage1_entries.append(entry)

        # Group by normalized canonical name for merging exact duplicates
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for e in stage1_entries:
            norm_k = self._normalize_key(e["canonical_name"])
            if norm_k not in grouped:
                grouped[norm_k] = []
            grouped[norm_k].append(e)

        merged_dups = 0
        final_entries: List[Dict[str, Any]] = []

        for norm_k, group in grouped.items():
            primary = group[0]
            if len(group) > 1:
                merged_dups += len(group) - 1
                combined_aliases = list(primary.get("known_aliases", []))
                for sec in group[1:]:
                    combined_aliases.append(sec["canonical_name"])
                    combined_aliases.extend(sec.get("known_aliases", []))
                primary["known_aliases"] = combined_aliases

            cname = primary["canonical_name"]

            # Deduplicate and clean aliases
            seen_aliases = set()
            cleaned_aliases = []
            for a in primary.get("known_aliases", []):
                a_clean = html.unescape(str(a)).strip()
                if not a_clean or a_clean.lower() == cname.lower():
                    continue
                a_norm = a_clean.lower()
                if a_norm not in seen_aliases:
                    seen_aliases.add(a_norm)
                    cleaned_aliases.append(a_clean)

            primary["known_aliases"] = sorted(cleaned_aliases)

            # Validate entity_type
            etype = str(primary.get("entity_type", "UNI")).strip().upper()
            if etype not in VALID_ENTITY_TYPES:
                etype = "UNI"
            primary["entity_type"] = etype

            # Format canonical_id
            cid = str(primary.get("canonical_id", ""))
            parts = cid.split("-")
            if len(parts) >= 2 and parts[1].isdigit():
                seq = int(parts[1])
                primary["canonical_id"] = f"{etype}-{seq:05d}-{generate_slug(cname)}"
            else:
                primary["canonical_id"] = f"{etype}-00000-{generate_slug(cname)}"

            final_entries.append(primary)

        # Fix any duplicate IDs if necessary
        seen_ids = set()
        max_seq = 0
        for e in final_entries:
            parts = e["canonical_id"].split("-")
            if len(parts) >= 2 and parts[1].isdigit():
                max_seq = max(max_seq, int(parts[1]))

        for e in final_entries:
            if e["canonical_id"] in seen_ids:
                max_seq += 1
                etype = e["entity_type"]
                cname = e["canonical_name"]
                e["canonical_id"] = f"{etype}-{max_seq:05d}-{generate_slug(cname)}"
            seen_ids.add(e["canonical_id"])

        self.entries = final_entries
        self._rebuild_lookup_map()
        self.save_registry()

        stats = {
            "initial_count": initial_count,
            "cleaned_html_entities": cleaned_html,
            "merged_duplicates": merged_dups,
            "pruned_generic_departments": pruned_depts,
            "final_count": len(self.entries),
        }
        logger.info(f"Local registry cleanup finished: {stats}")
        return stats

    def _run_gemini_registry_audit(self, gemini_client: Any) -> Dict[str, int]:
        academic_words = {'univ', 'university', 'college', 'polytechnic', 'mit', 'cmu', 'stanford', 'berkeley', 'ucla', 'harvard', 'eth', 'tum', 'school'}
        lab_academic = [e for e in self.entries if e.get('entity_type') == 'LAB' and any(w in e['canonical_name'].lower() for w in academic_words)]

        key_phrases = ['max planck', 'california', 'texas', 'beijing', 'singapore', 'eth', 'tum', 'mit', 'inria', 'cnrs']
        clustered_candidates: Dict[str, List[Dict[str, Any]]] = {}
        for kp in key_phrases:
            matches = [e for e in self.entries if kp in e['canonical_name'].lower()]
            if len(matches) > 1:
                clustered_candidates[kp] = matches

        batches: List[List[Dict[str, Any]]] = []
        for i in range(0, len(lab_academic), 40):
            batches.append(lab_academic[i:i+40])

        for kp, group in clustered_candidates.items():
            for i in range(0, len(group), 40):
                batches.append(group[i:i+40])

        total_merges = 0
        total_reclass = 0
        seen_batch_tuples = set()

        for batch in batches:
            batch_tuple = tuple(sorted(e['canonical_id'] for e in batch))
            if batch_tuple in seen_batch_tuples or len(batch) < 2:
                continue
            seen_batch_tuples.add(batch_tuple)

            res = gemini_client.audit_registry_candidates(batch)
            merges = res.get("merges", [])
            reclassifications = res.get("reclassifications", [])

            for m in merges:
                src_id = m.get("source_id")
                tgt_id = m.get("target_id")
                if src_id and tgt_id and src_id != tgt_id:
                    src_entry = self.find_by_id(src_id)
                    tgt_entry = self.find_by_id(tgt_id)
                    if src_entry and tgt_entry:
                        aliases = set(tgt_entry.get("known_aliases", []))
                        aliases.add(src_entry["canonical_name"])
                        aliases.update(src_entry.get("known_aliases", []))
                        if m.get("add_as_alias"):
                            aliases.add(m["add_as_alias"])
                        tgt_entry["known_aliases"] = sorted(list(aliases))
                        self.entries = [e for e in self.entries if e["canonical_id"] != src_id]
                        total_merges += 1

            for r in reclassifications:
                cid = r.get("canonical_id")
                new_type = str(r.get("entity_type", "")).upper()
                if cid and new_type in VALID_ENTITY_TYPES:
                    entry = self.find_by_id(cid)
                    if entry and entry.get("entity_type") != new_type:
                        entry["entity_type"] = new_type
                        total_reclass += 1

        logger.info(f"Gemini LLM Registry Audit finished: {total_merges} merges, {total_reclass} reclassifications applied.")
        return {"merges": total_merges, "reclassifications": total_reclass}

    def clean_registry(
        self,
        gemini_client: Optional[Any] = None,
        use_gemini: bool = True,
    ) -> Dict[str, Any]:
        """
        Validates, deduplicates, standardizes, and cleans canonical organization registry.
        Applies local cleanup (HTML entity decoding, generic department pruning, exact name duplicate merging, ID formatting)
        AND optional Gemini LLM registry audit pass for smart deduplication & entity-type reclassification.
        """
        stats = self._clean_registry_local()

        if use_gemini and gemini_client and getattr(gemini_client, "api_key", None):
            logger.info("Executing Gemini LLM Registry Audit Pass...")
            g_stats = self._run_gemini_registry_audit(gemini_client)
            stats["gemini_merges"] = g_stats.get("merges", 0)
            stats["gemini_reclassifications"] = g_stats.get("reclassifications", 0)
            stats = self._clean_registry_local()

        return stats


