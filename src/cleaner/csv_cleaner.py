import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import CLEANED_OUTPUT_DATA_DIR, DEFAULT_GEMINI_MODEL, OUTPUT_DATA_DIR
from src.normalizer.gemini_client import GeminiClient, parse_gemini_json
from src.registry.organization_registry import OrganizationRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_PASS1_TRIAGE = """You are an expert academic metadata triage assistant.
Your task is to review a list of raw institution names and identify ALL entries that have potential data quality issues:
1. DUPLICATES / VARIANTS: Multiple entries referring to the same parent institution (e.g., 'Google Brain' & 'Google LLC', 'MIT CSAIL' & 'MIT').
1. DUPLICATES / VARIANTS: Multiple entries referring to the same parent institution (e.g., 'Google Brain' & 'Google LLC', 'MIT CSAIL' & 'MIT', 'UCLA Vision Lab' & 'University of California, Los Angeles').
2. ONES TO BE PRUNED: Standalone department/faculty names lacking parent institutions (e.g., 'Department of Computer Science', 'Faculty of Engineering') or generic noise ('UNKNOWN', 'N/A', 'Independent Researcher').
3. MIX-UPS / HIERARCHY ISSUES: Entries with potential system vs campus confusion (e.g., 'University of California' vs 'University of California, San Diego') or miscategorized entities.
3. MIX-UPS / HIERARCHY ISSUES: Entries with potential system vs campus confusion (e.g., 'University of California' vs 'University of California, San Diego'), university labs misclassified as standalone LAB entities, or miscategorized entities.

INSTRUCTIONS:
Return ONLY a valid JSON object containing a list of problematic name strings from the input:
{
  "problematic_names": [
    "Department of Computer Science",
    "Google Brain",
    "University of California"
    "University of California",
    "UCLA Vision Lab"
  ]
}
"""

SYSTEM_PROMPT_PASS2_DEEP_ANALYSIS = """You are an expert academic metadata cleaning assistant.
You will be provided with a list of problematic institutional entries along with their current canonical_id, canonical_name, entity_type, and known merged aliases.

INSTITUTIONAL HIERARCHY RULES:
1. UNIVERSITIES:
   - Distinct campuses remain distinct entities (e.g., 'University of California, Los Angeles' vs. 'University of California, San Diego'). Campuses are NOT rolled up into parent university systems.
   - Internal university departments, faculties, labs, and research centers (e.g., 'UCLA Vision Lab', 'MIT CSAIL', 'Department of Robotics') MUST be merged into their parent university campus entity (e.g., 'University of California, Los Angeles', 'Massachusetts Institute of Technology') and assigned entity_type 'UNI'.
2. COMPANIES:
   - Global, regional, or departmental subsidiaries aggregate into a single parent entity (e.g., Google India, Google Brain, Google Zurich -> 'Google LLC'; FAIR -> 'Meta Platforms, Inc.').
3. LABS & GOVERNMENT AGENCIES:
   - Independent research institutes and national labs remain distinct entities (e.g., Max Planck Institutes, NASA Jet Propulsion Laboratory, CNRS).
   - Truly independent research institutes, government agencies, and national labs remain distinct entities (e.g., Max Planck Institutes, NASA Jet Propulsion Laboratory, CNRS, SRI International). University-affiliated labs are merged into their university campus as UNI.

INSTRUCTIONS:
Analyze the problematic entries and determine the cleaning actions:
1. PRUNING: Identify entries to prune (standalone department names without parent institution, generic noise like 'UNKNOWN', 'N/A').
2. MERGING: Group sub-entities, department variants, or corporate subsidiaries under their official parent canonical institution name.
3. CATEGORIZATION FIXES: Correct any miscategorized entity types (entity_type must be one of ['UNI', 'COM', 'LAB', 'GOV']). E.g. if an academic university was wrongly marked 'GOV' or 'COM', correct it to 'UNI'.
1. PRUNING: Identify entries to prune (standalone generic department names without parent institution e.g. 'Department of Computer Science' with no university specified, or generic noise like 'UNKNOWN', 'N/A').
2. MERGING: Group sub-entities, university labs/departments, or corporate subsidiaries under their official parent canonical institution name.
3. CATEGORIZATION FIXES: Correct any miscategorized entity types (entity_type must be one of ['UNI', 'COM', 'LAB', 'GOV']). E.g. if an academic university or university lab was marked 'LAB', 'GOV', or 'COM', correct it to 'UNI'.

OUTPUT FORMAT:
Return ONLY a valid JSON object:
{
  "prune_ids": ["ROW_ID_OR_NAME_1"],
  "merges": [
    {
      "target_parent_name": "Official Parent Organization Name",
      "source_ids": ["ROW_ID_OR_NAME_A", "ROW_ID_OR_NAME_B"]
    }
  ],
  "type_fixes": {
    "ROW_ID_OR_NAME_X": "UNI"
  }
}
"""


class CSVCleaner:
    """
    Independent tool to clean exported matrix CSV files using a 2-pass Gemini LLM + Python workflow:
    - Pass 1 (Triage): Send plain list of institution names (no codes, no JSON) to Gemini to identify problematic entries (duplicates, prunes, mix-ups).
    - Pass 2 (Deep Analysis): Send rich context for identified problematic entries (canonical IDs, names, entity types, and known aliases) to Gemini along with institutional hierarchy rules to get recommended actions (prunes, merges, entity_type fixes).
    - Pass 3 (Python Execution): Deterministically merge/prune rows, fix categorization, and aggregate paper counts in Python.
    - Features full Pause/Resume checkpointing (.checkpoint_<filename>.json) and file-level skip/caching support.
    - Saves cleaned CSV files to data/cleaned_output/ (never overwrites data/output/).
    - Leaves canonical organization registry COMPLETELY UNTOUCHED.
    """

    def __init__(
        self,
        registry: Optional[OrganizationRegistry] = None,
        gemini_client: Optional[GeminiClient] = None,
        model: Optional[str] = None,
        output_dir: Path = CLEANED_OUTPUT_DATA_DIR,
    ):
        self.registry = registry or OrganizationRegistry()
        if gemini_client:
            self.gemini_client = gemini_client
            if model:
                self.gemini_client.model = model
        else:
            self.gemini_client = GeminiClient(model=model or DEFAULT_GEMINI_MODEL)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _query_pass1_chunk(
        self, name_chunk: List[str], config: Any
    ) -> List[str]:
        """Pass 1: Queries Gemini with plain list of institution names to identify problematic entries."""
        if not name_chunk:
            return []

        formatted_list = "\n".join(f"- {name}" for name in name_chunk)
        contents_pass1 = [
            SYSTEM_PROMPT_PASS1_TRIAGE,
            f"RAW INSTITUTION NAMES:\n{formatted_list}",
        ]

        self.gemini_client.rate_limiter.acquire()
        try:
            response_pass1 = self.gemini_client.client.models.generate_content(
                model=self.gemini_client.model,
                contents=contents_pass1,
                config=config,
            )
            text_pass1 = self.gemini_client._extract_response_text(response_pass1)
            pass1_json = parse_gemini_json(text_pass1)
            prob_names = pass1_json.get("problematic_names", [])
            return prob_names if isinstance(prob_names, list) else []
        except Exception as e:
            err_str = str(e)
            is_timeout = "504" in err_str or "DEADLINE_EXCEEDED" in err_str or "timed out" in err_str.lower() or "timeout" in err_str.lower()
            is_json_err = isinstance(e, (json.JSONDecodeError, ValueError)) or "JSONDecodeError" in err_str or "Expecting property name" in err_str or "Expecting value" in err_str

            if (is_timeout or is_json_err) and len(name_chunk) > 30:
                mid = len(name_chunk) // 2
                reason = "timed out (504)" if is_timeout else "returned malformed JSON"
                logger.warning(
                    f"Pass 1 triage {reason} on chunk of {len(name_chunk)} names. "
                    f"Splitting into sub-chunks of {mid} and {len(name_chunk) - mid} names."
                )
                p1 = self._query_pass1_chunk(name_chunk[:mid], config)
                p2 = self._query_pass1_chunk(name_chunk[mid:], config)
                return p1 + p2
            logger.warning(f"Error querying Gemini LLM for Pass 1 triage chunk of {len(name_chunk)} names: {e}")
            return []

    def _query_pass2_deep_analysis(
        self, problematic_entries: List[Dict[str, Any]], config: Any
    ) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, str]]:
        """Pass 2: Queries Gemini with rich context (aliases, IDs, entity types) & rules for deep analysis."""
        if not problematic_entries:
            return [], [], {}

        payload_pass2 = {"problematic_entries": problematic_entries}
        contents_pass2 = [
            SYSTEM_PROMPT_PASS2_DEEP_ANALYSIS,
            f"INPUT PAYLOAD:\n{json.dumps(payload_pass2, separators=(',', ':'), ensure_ascii=False)}",
        ]

        self.gemini_client.rate_limiter.acquire()
        try:
            response_pass2 = self.gemini_client.client.models.generate_content(
                model=self.gemini_client.model,
                contents=contents_pass2,
                config=config,
            )
            text_pass2 = self.gemini_client._extract_response_text(response_pass2)
            pass2_json = parse_gemini_json(text_pass2)

            p_ids = pass2_json.get("prune_ids", [])
            m_groups = pass2_json.get("merges", [])
            t_fixes = pass2_json.get("type_fixes", {})

            return (
                p_ids if isinstance(p_ids, list) else [],
                m_groups if isinstance(m_groups, list) else [],
                t_fixes if isinstance(t_fixes, dict) else {},
            )
        except Exception as e:
            err_str = str(e)
            is_timeout = "504" in err_str or "DEADLINE_EXCEEDED" in err_str or "timed out" in err_str.lower() or "timeout" in err_str.lower()
            is_json_err = isinstance(e, (json.JSONDecodeError, ValueError)) or "JSONDecodeError" in err_str or "Expecting property name" in err_str or "Expecting value" in err_str

            if (is_timeout or is_json_err) and len(problematic_entries) > 15:
                mid = len(problematic_entries) // 2
                reason = "timed out (504)" if is_timeout else "returned malformed JSON"
                logger.warning(
                    f"Pass 2 analysis {reason} on chunk of {len(problematic_entries)} entries. "
                    f"Splitting into sub-chunks of {mid} and {len(problematic_entries) - mid} entries."
                )
                p1, m1, t1 = self._query_pass2_deep_analysis(problematic_entries[:mid], config)
                p2, m2, t2 = self._query_pass2_deep_analysis(problematic_entries[mid:], config)
                combined_t = {}
                combined_t.update(t1)
                combined_t.update(t2)
                return p1 + p2, m1 + m2, combined_t
            logger.warning(f"Error querying Gemini LLM for Pass 2 analysis of {len(problematic_entries)} entries: {e}")
            return [], [], {}

    def clean_file(
        self,
        input_csv_path: Path,
        output_csv_path: Optional[Path] = None,
        force: bool = False,
        refine: bool = False,
        max_passes: int = 3,
    ) -> Path:
        """
        Cleans a matrix CSV file recursively using a 2-pass workflow until Gemini suggests no further changes:
        1. If output file exists and neither force nor refine is set, skip.
        2. Determine starting file: existing cleaned file if refine=True, else raw input file.
        3. Iterate Pass 1 (Triage) -> Pass 2 (Deep Analysis) -> Pass 3 (Python Aggregation).
        4. If changes occur, save to output file and re-run using that cleaned file as starting point.
        5. Stop when Gemini proposes no further changes or max_passes is reached.
        """
        input_path = Path(input_csv_path)
        if not input_path.exists():
            err_msg = f"Input CSV file not found at {input_path}"
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        if not output_csv_path:
            output_csv_path = self.output_dir / input_path.name

        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine start file and handle skip logic
        if output_path.exists():
            if not force and not refine:
                logger.info(
                    f"Cleaned matrix CSV already exists at {output_path.resolve()}. "
                    f"Skipping LLM cleaning pass (use --force or --refine to re-clean)."
                )
                return output_path
            elif refine and not force:
                logger.info(f"Refining existing cleaned matrix CSV at {output_path.resolve()} recursively...")
                current_file = output_path
            else:
                logger.info(f"Force cleaning matrix CSV from raw source at {input_path.resolve()} recursively...")
                current_file = input_path
        else:
            logger.info(f"Starting CSV cleaning for {input_path.resolve()} -> {output_path.resolve()}")
            current_file = input_path

        checkpoint_path = output_path.parent / f".checkpoint_{input_path.name}.json"

        def load_checkpoint() -> Dict[str, Any]:
            if checkpoint_path.exists():
                try:
                    with open(checkpoint_path, "r", encoding="utf-8") as cp_file:
                        return json.load(cp_file)
                except Exception as e:
                    logger.warning(f"Could not read existing checkpoint at {checkpoint_path}: {e}")
            return {}

        def save_checkpoint(cp_data: Dict[str, Any]) -> None:
            try:
                with open(checkpoint_path, "w", encoding="utf-8") as cp_file:
                    json.dump(cp_data, cp_file, ensure_ascii=False)
            except Exception as cp_err:
                logger.warning(f"Could not save checkpoint to {checkpoint_path}: {cp_err}")

        def clear_checkpoint() -> None:
            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not remove checkpoint file {checkpoint_path}: {e}")

        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            should_return_http_response=True,
        )

        pass_num = 1
        checkpoint_data = load_checkpoint()

        while pass_num <= max_passes:
            logger.info(f"=== Recursive Cleaning Pass {pass_num} for {input_path.name} (Source: {current_file.name}) ===")

            with open(current_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)

            if not rows:
                logger.warning(f"CSV file {current_file} is empty. Writing empty file to {output_path}")
                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                clear_checkpoint()
                return output_path

            meta_fields = {"canonical_id", "canonical_name", "entity_type", "total"}
            year_fields = [fn for fn in fieldnames if fn not in meta_fields]

            if not self.gemini_client.api_key:
                logger.warning(f"GEMINI_API_KEY missing. Copying {current_file} to {output_path} without LLM cleaning.")
                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                clear_checkpoint()
                return output_path

            # PASS 1: Lightweight Triage
            raw_names = [r.get("canonical_name", "").strip() for r in rows if r.get("canonical_name", "").strip()]
            name_chunk_size = 150
            name_chunks = [raw_names[i : i + name_chunk_size] for i in range(0, len(raw_names), name_chunk_size)]

            all_problematic_names: Set[str] = set(checkpoint_data.get("all_problematic_names", []))
            pass1_val = checkpoint_data.get("pass1_completed_chunks", 0)
            pass1_start_chunk = len(pass1_val) if isinstance(pass1_val, list) else int(pass1_val)

            if pass1_start_chunk < len(name_chunks):
                logger.info(f"Pass 1 (Triage): Querying Gemini LLM with {len(raw_names)} names across {len(name_chunks)} chunk(s) (resuming from chunk {pass1_start_chunk + 1})...")
                for idx in range(pass1_start_chunk, len(name_chunks)):
                    chunk = name_chunks[idx]
                    logger.info(f"Pass 1 (Chunk {idx + 1}/{len(name_chunks)}): Triaging {len(chunk)} plain institution names...")
                    prob = self._query_pass1_chunk(chunk, config)
                    for p in prob:
                        if str(p).strip():
                            all_problematic_names.add(str(p).strip().lower())

                    checkpoint_data["pass1_completed_chunks"] = idx + 1
                    checkpoint_data["all_problematic_names"] = sorted(list(all_problematic_names))
                    save_checkpoint(checkpoint_data)

            logger.info(f"Pass 1 Complete: Identified {len(all_problematic_names)} potential problematic name entries.")

            if not all_problematic_names:
                logger.info(f"Gemini suggested no further problematic entries on pass {pass_num}. Recursive cleaning converged!")
                if current_file != output_path or not output_path.exists():
                    with open(output_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                clear_checkpoint()
                return output_path

            # Prepare problematic entries payload for Pass 2
            problematic_entries_payload: List[Dict[str, Any]] = []
            for r in rows:
                c_id = (r.get("canonical_id") or "").strip()
                c_name = (r.get("canonical_name") or "").strip()
                e_type = (r.get("entity_type") or "UNI").strip()

                if c_name.lower() in all_problematic_names or c_id.lower() in all_problematic_names:
                    aliases = []
                    if c_id:
                        reg_entry = self.registry._lookup_map.get(c_id.upper()) or self.registry.find_by_string(c_id)
                        if reg_entry:
                            aliases = reg_entry.get("known_aliases", [])

                    problematic_entries_payload.append({
                        "canonical_id": c_id,
                        "canonical_name": c_name,
                        "entity_type": e_type,
                        "known_aliases": aliases,
                    })

            # PASS 2: Targeted Deep Analysis
            entry_chunk_size = 80
            entry_chunks = [problematic_entries_payload[i : i + entry_chunk_size] for i in range(0, len(problematic_entries_payload), entry_chunk_size)]

            all_prune_ids: List[str] = checkpoint_data.get("all_prune_ids", [])
            all_merges: List[Dict[str, Any]] = checkpoint_data.get("all_merges", [])
            all_type_fixes: Dict[str, str] = checkpoint_data.get("all_type_fixes", {})

            pass2_val = checkpoint_data.get("pass2_completed_chunks", 0)
            pass2_start_chunk = len(pass2_val) if isinstance(pass2_val, list) else int(pass2_val)

            if pass2_start_chunk < len(entry_chunks):
                logger.info(f"Pass 2 (Deep Analysis): Analyzing {len(problematic_entries_payload)} entries across {len(entry_chunks)} chunk(s)...")
                for idx in range(pass2_start_chunk, len(entry_chunks)):
                    chunk = entry_chunks[idx]
                    logger.info(f"Pass 2 (Chunk {idx + 1}/{len(entry_chunks)}): Deep analyzing {len(chunk)} entries...")
                    p_ids, m_groups, t_fixes = self._query_pass2_deep_analysis(chunk, config)
                    all_prune_ids.extend(p_ids)
                    all_merges.extend(m_groups)
                    all_type_fixes.update(t_fixes)

                    checkpoint_data["pass2_completed_chunks"] = idx + 1
                    checkpoint_data["all_prune_ids"] = all_prune_ids
                    checkpoint_data["all_merges"] = all_merges
                    checkpoint_data["all_type_fixes"] = all_type_fixes
                    save_checkpoint(checkpoint_data)

            if not all_prune_ids and not all_merges and not all_type_fixes:
                logger.info(f"Gemini suggested no further cleaning actions (prunes/merges/type fixes) on pass {pass_num}. Recursive cleaning converged!")
                if current_file != output_path or not output_path.exists():
                    with open(output_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                clear_checkpoint()
                return output_path

            # PASS 3: Local Registry Target Mapping & Deterministic Python Execution
            prune_set: Set[str] = {str(pid).strip().lower() for pid in all_prune_ids if pid}
            source_id_to_target: Dict[str, Tuple[str, str, str]] = {}

            for m in all_merges:
                if not isinstance(m, dict):
                    continue
                parent_name = (m.get("target_parent_name") or "").strip()
                sources = m.get("source_ids") or []
                if not parent_name or not sources:
                    continue

                match = self.registry.find_by_string(parent_name)
                if match:
                    c_id = match.get("canonical_id", "")
                    c_name = match.get("canonical_name", parent_name)
                    e_type = match.get("entity_type", "UNI")
                else:
                    c_id = "UNI-00000-CUSTOM"
                    c_name = parent_name
                    e_type = "UNI"

                target_tuple = (c_id, c_name, e_type)
                for sid in sources:
                    sid_clean = str(sid).strip().lower()
                    if sid_clean:
                        source_id_to_target[sid_clean] = target_tuple

            cleaned_map: Dict[Tuple[str, str, str], Dict[str, int]] = {}
            unmerged_rows: List[Dict[str, Any]] = []
            type_fixes_map: Dict[str, str] = {str(k).strip().lower(): str(v).strip().upper() for k, v in all_type_fixes.items()}

            for r in rows:
                raw_id = (r.get("canonical_id") or "").strip()
                raw_name = (r.get("canonical_name") or "").strip()
                raw_type = (r.get("entity_type") or "UNI").strip()

                if raw_id.lower() in type_fixes_map:
                    raw_type = type_fixes_map[raw_id.lower()]
                    new_type = type_fixes_map[raw_id.lower()]
                    if new_type != raw_type:
                        logger.info(f"Pass {pass_num}: Categorization fix for '{raw_name}' ({raw_id}): {raw_type} -> {new_type}")
                    raw_type = new_type
                elif raw_name.lower() in type_fixes_map:
                    raw_type = type_fixes_map[raw_name.lower()]
                    new_type = type_fixes_map[raw_name.lower()]
                    if new_type != raw_type:
                        logger.info(f"Pass {pass_num}: Categorization fix for '{raw_name}' ({raw_id}): {raw_type} -> {new_type}")
                    raw_type = new_type

                if raw_id.lower() in prune_set or raw_name.lower() in prune_set:
                    logger.info(f"Pass {pass_num}: Pruning row '{raw_name}' ({raw_id})")
                    continue

                target_key = source_id_to_target.get(raw_id.lower()) or source_id_to_target.get(raw_name.lower())

                if target_key:
                    c_id, c_name, c_type = target_key
                    if c_id.lower() in type_fixes_map:
                        c_type = type_fixes_map[c_id.lower()]
                    elif c_name.lower() in type_fixes_map:
                        c_type = type_fixes_map[c_name.lower()]

                    logger.info(f"Pass {pass_num}: Merging row '{raw_name}' ({raw_id}) -> '{c_name}' ({c_id}, {c_type})")

                    final_key = (c_id, c_name, c_type)
                    if final_key not in cleaned_map:
                        cleaned_map[final_key] = {y: 0 for y in year_fields}

                    for y in year_fields:
                        val = r.get(y, 0)
                        cleaned_map[final_key][y] += int(val) if str(val).isdigit() else 0
                else:
                    row_copy = dict(r)
                    row_copy["entity_type"] = raw_type
                    unmerged_rows.append(row_copy)

            final_output_rows: List[Dict[str, Any]] = []
            for (c_id, c_name, e_type), year_counts in cleaned_map.items():
                row_dict = {"canonical_id": c_id, "canonical_name": c_name, "entity_type": e_type}
                tot = 0
                for y in year_fields:
                    cnt = year_counts[y]
                    row_dict[y] = cnt
                    tot += cnt
                row_dict["total"] = tot
                if tot > 0:
                    final_output_rows.append(row_dict)

            for r in unmerged_rows:
                c_id = r.get("canonical_id", "")
                c_name = r.get("canonical_name", "")
                e_type = r.get("entity_type", "UNI")
                row_dict = {"canonical_id": c_id, "canonical_name": c_name, "entity_type": e_type}
                tot = 0
                for y in year_fields:
                    val = r.get(y, 0)
                    cnt = int(val) if str(val).isdigit() else 0
                    row_dict[y] = cnt
                    tot += cnt
                row_dict["total"] = tot
                if tot > 0:
                    final_output_rows.append(row_dict)

            final_output_rows.sort(key=lambda r: (-int(r.get("total", 0)), str(r.get("canonical_name", ""))))

            if len(final_output_rows) == len(rows) and final_output_rows == rows:
                logger.info(f"Pass {pass_num}: LLM suggestions resulted in no effective row modifications. Recursive cleaning converged!")
                if current_file != output_path or not output_path.exists():
                    with open(output_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(final_output_rows)
                clear_checkpoint()
                return output_path

            # Save intermediate cleaned CSV to output_path
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(final_output_rows)

            pruned_count = len(rows) - len(final_output_rows)
            logger.info(f"Pass {pass_num} Complete for {input_path.name}: {len(rows)} rows -> {len(final_output_rows)} rows (pruned/combined {pruned_count}). Saved to {output_path}")

            # Re-run taking newly written cleaned file as starting point for next pass iteration
            current_file = output_path
            checkpoint_data = {}
            clear_checkpoint()
            pass_num += 1

        logger.info(f"Reached maximum recursive cleaning passes ({max_passes}) for {input_path.name}.")
        return output_path

    def clean_all(
        self,
        input_dir: Path = OUTPUT_DATA_DIR,
        output_dir: Optional[Path] = None,
        force: bool = False,
        refine: bool = False,
        batch_config: Optional[Any] = None,
        config_path: Optional[Path] = None,
        max_passes: int = 3,
    ) -> List[Path]:
        """Cleans all CSV matrix files in input_dir (obeying batch.yaml if present) and saves them to output_dir."""
        in_dir = Path(input_dir)
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        eff_config = batch_config
        target_config_name = config_path.name if config_path else "batch.yaml"
        if not eff_config:
            target_config = config_path if config_path else Path("batch.yaml")
            if target_config.exists():
                try:
                    from src.runner.batch_config import BatchConfig
                    eff_config = BatchConfig.from_file(target_config)
                    target_config_name = target_config.name
                    logger.info(f"Obeying batch configuration from {target_config.resolve()} for CSV cleaning ({len(eff_config.venues)} active venues)")
                except Exception as e:
                    logger.warning(f"Could not load batch configuration from {target_config}: {e}")

        allowed_venues: Optional[Set[str]] = None
        if eff_config:
            from src.utils import sanitize_venue_name
            allowed_venues = set()
            for v_cfg in eff_config.venue_configs:
                allowed_venues.add(v_cfg.short_name.upper())
                if isinstance(v_cfg.search_term, str):
                    allowed_venues.add(sanitize_venue_name(v_cfg.search_term).upper())
                elif isinstance(v_cfg.search_term, list):
                    for st in v_cfg.search_term:
                        allowed_venues.add(sanitize_venue_name(st).upper())

        cleaned_paths: List[Path] = []
        csv_files = sorted(list(in_dir.glob("*.csv")))
        if not csv_files:
            logger.info(f"No CSV files found in {in_dir} to clean.")
            return []

        import re
        for csv_file in csv_files:
            m = re.match(r"^([A-Za-z0-9_\-]+)_affiliations_(\d{4})_(\d{4})\.csv$", csv_file.name)
            if m:
                v_code = m.group(1).upper()
                from src.utils import sanitize_venue_name
                if allowed_venues is not None and v_code not in allowed_venues and sanitize_venue_name(v_code).upper() not in allowed_venues:
                    logger.info(f"Excluding CSV file '{csv_file.name}' from cleaning (venue '{v_code}' not active in {target_config_name}).")
                    continue

            target_out = out_dir / csv_file.name
            cleaned_path = self.clean_file(csv_file, output_csv_path=target_out, force=force, refine=refine, max_passes=max_passes)
            cleaned_paths.append(cleaned_path)

        return cleaned_paths
