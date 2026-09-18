import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config import CLEANED_OUTPUT_DATA_DIR, OUTPUT_DATA_DIR
from src.normalizer.gemini_client import GEMINI_MODEL, GeminiClient
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

SYSTEM_PROMPT_STEP3_CLEAN = """You are an expert academic metadata cleaning assistant.
Your task is to produce the final cleaned CSV matrix of institution paper counts across years based on targeted canonical registry entries and merge instructions.

INSTRUCTIONS:
1. Remove all rows listed in prune_ids.
2. Apply the requested merges: for each merge group, combine all source_ids into a single row using the canonical_id, canonical_name, and entity_type from the targeted registry entries provided (or clean parent name if not in registry). Sum up paper counts for every year.
3. Keep all other valid un-merged rows unchanged.

OUTPUT FORMAT:
Return ONLY a valid JSON object:
{
  "cleaned_rows": [
    {
      "canonical_id": "UNI-00001-STANFD",
      "canonical_name": "Stanford University",
      "entity_type": "UNI",
      "counts": {
        "2017": 5,
        "2018": 12
      }
    }
  ]
}
"""


class CSVCleaner:
    """
    Independent tool to clean exported matrix CSV files using a 3-step targeted Gemini LLM workflow.
    - Step 1: Send CSV rows to Gemini (WITHOUT full canonical list) to identify prunes & request parent names (e.g. "MIT").
    - Step 2: Look up specific requested parent names locally in OrganizationRegistry to create a small targeted JSON context.
    - Step 3: Send CSV rows + targeted registry context to Gemini to generate the final cleaned matrix.
    - Saves cleaned CSV files to data/cleaned_output/ (never overwrites data/output/).
    - Leaves canonical organization registry COMPLETELY UNTOUCHED.
    """

    def __init__(
        self,
        registry: Optional[OrganizationRegistry] = None,
        gemini_client: Optional[GeminiClient] = None,
        output_dir: Path = CLEANED_OUTPUT_DATA_DIR,
    ):
        self.registry = registry or OrganizationRegistry()
        self.gemini_client = gemini_client or GeminiClient()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def clean_file(
        self,
        input_csv_path: Path,
        output_csv_path: Optional[Path] = None,
    ) -> Path:
        """
        Cleans a matrix CSV file using a 3-step targeted Gemini LLM workflow:
        1. Query Gemini with CSV to get prune_ids and requested_parent_names (e.g. "MIT").
        2. Look up requested_parent_names locally in OrganizationRegistry.
        3. Query Gemini with CSV + targeted registry entries to produce the final cleaned matrix.
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

        logger.info(f"Starting 3-step targeted CSV cleaning for {input_path.resolve()} -> {output_path.resolve()}")

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

        # Format CSV rows compactly for Gemini (NO full canonical list sent)
        compact_rows = []
        for r in rows:
            c_id = r.get("canonical_id", "")
            c_name = r.get("canonical_name", "")
            e_type = r.get("entity_type", "")
            counts = {y: int(r[y]) for y in year_fields if y in r and str(r[y]).isdigit() and int(r[y]) > 0}
            compact_rows.append({
                "canonical_id": c_id,
                "canonical_name": c_name,
                "entity_type": e_type,
                "counts": counts,
            })

        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            should_return_http_response=True,
        )

        # STEP 1: Ask Gemini to analyze CSV, identify prunes, and request parent names for merges
        logger.info(f"Step 1: Sending CSV ({len(rows)} rows) to Gemini to identify prunes & request parent merge names...")
        payload_step1 = {"matrix_rows": compact_rows}
        contents_step1 = [
            SYSTEM_PROMPT_STEP1_ANALYSIS,
            f"INPUT PAYLOAD:\n{json.dumps(payload_step1, separators=(',', ':'), ensure_ascii=False)}",
        ]

        self.gemini_client.rate_limiter.acquire()
        response_step1 = self.gemini_client.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents_step1,
            config=config,
        )

        text_step1 = self.gemini_client._extract_response_text(response_step1)
        step1_json = json.loads(text_step1)

        prune_ids: List[str] = step1_json.get("prune_ids", [])
        requested_parent_names: List[str] = step1_json.get("requested_parent_names", [])
        merges: List[Dict[str, Any]] = step1_json.get("merges", [])

        # Collect all requested parent names from step1 output
        all_requested_names: Set[str] = set(requested_parent_names)
        for m in merges:
            if isinstance(m, dict) and m.get("target_parent_name"):
                all_requested_names.add(m["target_parent_name"])

        logger.info(f"Step 1 Complete: Gemini requested {len(all_requested_names)} parent names for merges and identified {len(prune_ids)} rows to prune.")

        # STEP 2: Look up specific requested parent names in local OrganizationRegistry
        targeted_registry: Dict[str, Any] = {}
        for parent_name in sorted(list(all_requested_names)):
            match = self.registry.find_by_string(parent_name)
            if match:
                targeted_registry[parent_name] = {
                    "canonical_id": match.get("canonical_id"),
                    "canonical_name": match.get("canonical_name"),
                    "entity_type": match.get("entity_type"),
                    "known_aliases": match.get("known_aliases", []),
                }
                logger.info(f"Step 2: Registry matched requested parent '{parent_name}' -> {match.get('canonical_id')} ({match.get('canonical_name')})")
            else:
                targeted_registry[parent_name] = {
                    "canonical_id": None,
                    "canonical_name": parent_name,
                    "entity_type": "UNI",
                }
                logger.info(f"Step 2: Requested parent '{parent_name}' not in registry. Using clean name.")

        # STEP 3: Send CSV + targeted registry snippet + merge instructions to Gemini for final matrix generation
        logger.info(f"Step 3: Sending CSV + targeted registry snippet ({len(targeted_registry)} parent entries) to Gemini for final clean matrix generation...")
        payload_step3 = {
            "prune_ids": prune_ids,
            "merges": merges,
            "targeted_registry_entries": targeted_registry,
            "matrix_rows": compact_rows,
        }

        contents_step3 = [
            SYSTEM_PROMPT_STEP3_CLEAN,
            f"INPUT PAYLOAD:\n{json.dumps(payload_step3, separators=(',', ':'), ensure_ascii=False)}",
        ]

        self.gemini_client.rate_limiter.acquire()
        response_step3 = self.gemini_client.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents_step3,
            config=config,
        )

        text_step3 = self.gemini_client._extract_response_text(response_step3)
        step3_json = json.loads(text_step3)
        cleaned_list = step3_json.get("cleaned_rows") or step3_json.get("cleaned_matrix") or []

        # Process cleaned rows and re-calculate exact totals in Python
        cleaned_output_rows: List[Dict[str, Any]] = []

        for item in cleaned_list:
            if not isinstance(item, dict):
                continue
            c_id = (item.get("canonical_id") or "").strip()
            c_name = (item.get("canonical_name") or "").strip()
            e_type = (item.get("entity_type") or "UNI").strip()
            item_counts = item.get("counts") or {}

            if not c_name:
                continue

            row_dict = {
                "canonical_id": c_id or "UNI-00000-CUSTOM",
                "canonical_name": c_name,
                "entity_type": e_type,
            }

            total = 0
            for y in year_fields:
                cnt = int(item_counts.get(y, 0)) if str(item_counts.get(y, 0)).isdigit() else 0
                row_dict[y] = cnt
                total += cnt

            row_dict["total"] = total
            if total > 0:
                cleaned_output_rows.append(row_dict)

        # Fallback if LLM produced 0 rows: preserve original rows
        if not cleaned_output_rows:
            logger.warning("LLM produced 0 cleaned rows. Falling back to original CSV rows.")
            cleaned_output_rows = rows

        # Sort descending by total, then by canonical_name
        cleaned_output_rows.sort(key=lambda r: (-int(r.get("total", 0)), str(r.get("canonical_name", ""))))

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_output_rows)

        pruned_count = len(rows) - len(cleaned_output_rows)
        logger.info(f"CSV Cleaning Complete for {input_path.name}: {len(rows)} input rows -> {len(cleaned_output_rows)} cleaned rows (pruned/combined {pruned_count} entries). Saved to {output_path}")

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
