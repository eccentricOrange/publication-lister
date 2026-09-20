import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import CLEANED_OUTPUT_DATA_DIR, DEFAULT_GEMINI_MODEL, OUTPUT_DATA_DIR
from src.normalizer.gemini_client import GeminiClient
from src.registry.organization_registry import OrganizationRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_STEP1_ANALYSIS = """You are an expert academic metadata and institution entity cleaning assistant.
Your task is to analyze a CSV matrix of academic paper affiliation counts across years and decide:
1. Which rows should be PRUNED (useless, unidentifiable, or standalone department names without parent organization).
2. Which rows should be MERGED/COMBINED (sub-entities, department variants, lab branches into their parent organization).

RULES:
1. PRUNE:
   - Identify standalone department/faculty names that lack a parent institution (e.g., 'Department of Electrical Engineering', 'Department of Computer Science', 'Faculty of Science', 'Dept of CS').
   - Identify generic, invalid, or meaningless rows like 'UNKNOWN', 'N/A', 'Unknown Organization', 'Independent Researcher', 'Various Institutions'.

2. MERGE / COMBINE:
   - Group sub-entities, research labs, or department variants into their proper parent canonical institution name.
   - For example: 'MIT CSAIL', 'MIT Media Labs', and 'Massachusetts Institute of Technology' MUST be grouped together, and you should request the parent institution name 'MIT'.
   - For example: 'Stanford AI Lab', 'Stanford CS', 'Stanford Vision Lab' -> group together and request parent name 'Stanford University'.

3. PRESERVE DISTINCT VALID ENTITIES:
   - Keep distinct university campuses separate (e.g., 'University of California, Berkeley' vs 'University of California, Los Angeles').
   - Keep distinct independent research labs separate (e.g., 'Max Planck Institute...', 'CNRS', 'NASA JPL').

OUTPUT FORMAT:
Return ONLY a valid JSON object:
{
  "prune_ids": ["ROW_ID_1", "ROW_ID_2"],
  "requested_parent_names": ["MIT", "Stanford University"],
  "merges": [
    {
      "target_parent_name": "MIT",
      "source_ids": ["UNI-00002-MITCAM", "UNI-00099-MITMED"]
    },
    {
      "target_parent_name": "Stanford University",
      "source_ids": ["UNI-00001-STANFD", "UNI-00055-STANLAB"]
    }
  ]
}
"""


class CSVCleaner:
    """
    Independent tool to clean exported matrix CSV files using a hybrid Gemini LLM + Python workflow:
    - Step 1: Query Gemini LLM with row names to identify prunes & request parent merge names (e.g. "MIT").
    - Step 2: Look up specific requested parent names locally in OrganizationRegistry (read-only).
    - Step 3: Deterministically merge/prune rows and aggregate yearly paper counts in Python (fast & fail-proof).
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

    def _query_step1_chunk(
        self, chunk: List[Dict[str, Any]], config: Any
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """Queries Gemini LLM for Step 1 analysis with adaptive sub-chunking on timeout."""
        if not chunk:
            return [], [], []

        payload_step1 = {"matrix_rows": chunk}
        contents_step1 = [
            SYSTEM_PROMPT_STEP1_ANALYSIS,
            f"INPUT PAYLOAD:\n{json.dumps(payload_step1, separators=(',', ':'), ensure_ascii=False)}",
        ]

        self.gemini_client.rate_limiter.acquire()
        try:
            response_step1 = self.gemini_client.client.models.generate_content(
                model=self.gemini_client.model,
                contents=contents_step1,
                config=config,
            )
            text_step1 = self.gemini_client._extract_response_text(response_step1)
            step1_json = json.loads(text_step1)

            p_ids = step1_json.get("prune_ids", [])
            p_names = step1_json.get("requested_parent_names", [])
            m_groups = step1_json.get("merges", [])

            return (
                p_ids if isinstance(p_ids, list) else [],
                p_names if isinstance(p_names, list) else [],
                m_groups if isinstance(m_groups, list) else [],
            )
        except Exception as e:
            err_str = str(e)
            is_timeout = "504" in err_str or "DEADLINE_EXCEEDED" in err_str or "timed out" in err_str.lower() or "timeout" in err_str.lower()
            if is_timeout and len(chunk) > 25:
                mid = len(chunk) // 2
                logger.warning(
                    f"Step 1 query timed out (504/Deadline Exceeded) on chunk of {len(chunk)} rows. "
                    f"Splitting into sub-chunks of {mid} and {len(chunk) - mid} rows."
                )
                p1, n1, m1 = self._query_step1_chunk(chunk[:mid], config)
                p2, n2, m2 = self._query_step1_chunk(chunk[mid:], config)
                return p1 + p2, n1 + n2, m1 + m2
            logger.warning(f"Error querying Gemini LLM for cleaning chunk of {len(chunk)} rows: {e}")
            return [], [], []

    def clean_file(
        self,
        input_csv_path: Path,
        output_csv_path: Optional[Path] = None,
    ) -> Path:
        """
        Cleans a matrix CSV file:
        1. Query Gemini LLM to get prune_ids and requested_parent_names (e.g. "MIT").
        2. Look up requested_parent_names locally in OrganizationRegistry to obtain canonical entity IDs & names.
        3. Deterministically aggregate paper counts and output cleaned CSV in Python.
        - Canonical registry is NOT modified.
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

        logger.info(f"Starting CSV cleaning for {input_path.resolve()} -> {output_path.resolve()}")

        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        if not rows:
            logger.warning(f"Input CSV {input_path} is empty. Writing empty file to {output_path}")
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            return output_path

        # Identify metadata fields and year fields
        meta_fields = {"canonical_id", "canonical_name", "entity_type", "total"}
        year_fields = [fn for fn in fieldnames if fn not in meta_fields]

        if not self.gemini_client.api_key:
            logger.warning(f"GEMINI_API_KEY missing. Copying original CSV {input_path} to {output_path} without LLM cleaning.")
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return output_path

        # Format CSV rows compactly for Gemini (NO counts or full canonical list sent)
        compact_rows = []
        for r in rows:
            c_id = r.get("canonical_id", "")
            c_name = r.get("canonical_name", "")
            e_type = r.get("entity_type", "")
            compact_rows.append({
                "canonical_id": c_id,
                "canonical_name": c_name,
                "entity_type": e_type,
            })

        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            should_return_http_response=True,
        )

        # STEP 1: Query Gemini to analyze CSV in chunks of max 100 rows to prevent 504 Gateway Timeouts
        chunk_size = 100
        row_chunks = [compact_rows[i : i + chunk_size] for i in range(0, len(compact_rows), chunk_size)]

        all_prune_ids: List[str] = []
        all_requested_parent_names: List[str] = []
        all_merges: List[Dict[str, Any]] = []

        logger.info(f"Step 1: Querying Gemini LLM for {len(rows)} matrix rows across {len(row_chunks)} chunk(s) (max {chunk_size} rows/chunk)...")

        for idx, chunk in enumerate(row_chunks, 1):
            logger.info(f"Step 1 (Chunk {idx}/{len(row_chunks)}): Querying Gemini LLM with {len(chunk)} matrix row headers...")
            p_ids, p_names, m_groups = self._query_step1_chunk(chunk, config)
            all_prune_ids.extend(p_ids)
            all_requested_parent_names.extend(p_names)
            all_merges.extend(m_groups)

        prune_ids: List[str] = all_prune_ids
        requested_parent_names: List[str] = list(set(all_requested_parent_names))
        merges: List[Dict[str, Any]] = all_merges

        prune_set: Set[str] = {str(pid).strip().lower() for pid in prune_ids if pid}

        # STEP 2: Look up requested parent names in local OrganizationRegistry to obtain canonical metadata
        source_id_to_target: Dict[str, Tuple[str, str, str]] = {}

        for m in merges:
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

            logger.info(f"Step 2: Mapped merge parent '{parent_name}' -> Canonical: {c_id} ({c_name}) for {len(sources)} source IDs")

        # STEP 3: Deterministically aggregate and prune in Python
        cleaned_map: Dict[Tuple[str, str, str], Dict[str, int]] = {}
        unmerged_rows: List[Dict[str, Any]] = []

        for r in rows:
            raw_id = (r.get("canonical_id") or "").strip()
            raw_name = (r.get("canonical_name") or "").strip()
            raw_type = (r.get("entity_type") or "UNI").strip()

            # Check if row should be pruned
            if raw_id.lower() in prune_set or raw_name.lower() in prune_set:
                logger.info(f"Pruning useless row: '{raw_name}' ({raw_id})")
                continue

            # Check if row belongs to a merge group
            target_key = source_id_to_target.get(raw_id.lower()) or source_id_to_target.get(raw_name.lower())

            if target_key:
                if target_key not in cleaned_map:
                    cleaned_map[target_key] = {y: 0 for y in year_fields}

                for y in year_fields:
                    val = r.get(y, 0)
                    cleaned_map[target_key][y] += int(val) if str(val).isdigit() else 0
            else:
                unmerged_rows.append(r)

        # Re-construct merged output rows
        final_output_rows: List[Dict[str, Any]] = []

        for (c_id, c_name, e_type), year_counts in cleaned_map.items():
            row_dict = {
                "canonical_id": c_id,
                "canonical_name": c_name,
                "entity_type": e_type,
            }
            tot = 0
            for y in year_fields:
                cnt = year_counts[y]
                row_dict[y] = cnt
                tot += cnt
            row_dict["total"] = tot
            if tot > 0:
                final_output_rows.append(row_dict)

        # Include unmerged valid rows
        for r in unmerged_rows:
            c_id = r.get("canonical_id", "")
            c_name = r.get("canonical_name", "")
            e_type = r.get("entity_type", "UNI")
            row_dict = {
                "canonical_id": c_id,
                "canonical_name": c_name,
                "entity_type": e_type,
            }
            tot = 0
            for y in year_fields:
                val = r.get(y, 0)
                cnt = int(val) if str(val).isdigit() else 0
                row_dict[y] = cnt
                tot += cnt
            row_dict["total"] = tot
            if tot > 0:
                final_output_rows.append(row_dict)

        # Sort descending by total, then by canonical_name
        final_output_rows.sort(key=lambda r: (-int(r.get("total", 0)), str(r.get("canonical_name", ""))))

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_output_rows)

        pruned_count = len(rows) - len(final_output_rows)
        logger.info(f"CSV Cleaning Complete for {input_path.name}: {len(rows)} input rows -> {len(final_output_rows)} cleaned rows (pruned/combined {pruned_count} entries). Saved to {output_path}")

        return output_path

    def clean_all(
        self,
        input_dir: Path = OUTPUT_DATA_DIR,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """Cleans all CSV matrix files in input_dir and saves them to output_dir."""
        in_dir = Path(input_dir)
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        cleaned_paths: List[Path] = []
        csv_files = sorted(list(in_dir.glob("*.csv")))
        if not csv_files:
            logger.info(f"No CSV files found in {in_dir} to clean.")
            return []

        for csv_file in csv_files:
            target_out = out_dir / csv_file.name
            cleaned_path = self.clean_file(csv_file, output_csv_path=target_out)
            cleaned_paths.append(cleaned_path)

        return cleaned_paths
