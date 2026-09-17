import json
import logging
import math
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

from src.config import GEMINI_API_KEY
from src.normalizer.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

SYSTEM_PROMPT = """You are an expert institutional entity normalization assistant for academic papers.
Your task is to map raw author affiliation strings to canonical organizations based on strict rules:

1. UNIVERSITY CAMPUS RULE:
   - Preserve distinct university campuses as separate entities.
   - Example: 'University of California, Los Angeles' and 'University of California, San Diego' MUST NOT be rolled up into 'University of California'. Keep them as distinct entities.

2. CORPORATE ROLLUP RULE:
   - Aggregate all regional, functional, or subsidiary corporate entities into a single parent entity.
   - Example: 'Google Brain', 'Google India', 'Google Zurich', 'Google Research' -> 'Google LLC'.
   - Example: 'FAIR', 'Facebook AI Research' -> 'Meta Platforms, Inc.'.

3. LABS & GOVERNMENT AGENCIES RULE:
   - Keep independent research labs and government agencies distinct (e.g., 'Max Planck Institute for Intelligent Systems', 'NASA Jet Propulsion Laboratory', 'CNRS').

INSTRUCTIONS:
For each raw affiliation string in the input list:
- Check if it matches an existing canonical entity from the provided registry.
- If it matches an existing entity, return its 'canonical_id'.
- If NO existing match fits, propose a NEW canonical entry with:
  - 'canonical_name': Formal, official name (e.g., 'University of Texas at Austin')
  - 'entity_type': One of ['UNI', 'COM', 'LAB', 'GOV']
  - 'known_aliases': List of aliases including the raw string

OUTPUT FORMAT:
Return ONLY a valid JSON object mapping each raw affiliation string to its resolution object:
{
  "resolutions": {
    "<raw_string>": {
      "canonical_id": "<existing_id_or_null>",
      "canonical_name": "<official_name>",
      "entity_type": "<UNI|COM|LAB|GOV>",
      "proposed_new": true_or_false
    }
  }
}
"""


class GeminiClient:
    """
    Gemini API Client for affiliation string normalization.
    Integrates TokenBucketRateLimiter, dynamic TPM/RPM limit discovery, and adaptive batch sizing.
    Strictly raises exceptions on missing API key or network failures.
    """

    def __init__(self, api_key: str = GEMINI_API_KEY, rate_limiter: Optional[TokenBucketRateLimiter] = None):
        self.api_key = api_key
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter(requests_per_minute=6.0, tokens_per_minute=250000.0)
        self.session = requests.Session()
        self.initial_rate_queried = False

    def _inspect_headers(self, response: requests.Response) -> None:
        """Inspects rate limit headers for RPM and TPM limits on initial query."""
        if self.initial_rate_queried:
            return
        headers = response.headers
        rpm_val = None
        tpm_val = None

        for key in ["x-ratelimit-limit-requests", "x-ratelimit-limit-rpm", "x-ratelimit-limit"]:
            if key in headers:
                try:
                    rpm_val = float(headers[key])
                    break
                except ValueError:
                    pass

        for key in ["x-ratelimit-limit-tokens", "x-ratelimit-limit-tpm"]:
            if key in headers:
                try:
                    tpm_val = float(headers[key])
                    break
                except ValueError:
                    pass

        if rpm_val or tpm_val:
            self.rate_limiter.update_limits(requests_per_minute=rpm_val, tokens_per_minute=tpm_val)
            self.initial_rate_queried = True

    def calculate_dynamic_batch(
        self,
        unresolved_strings: List[str],
        canonical_registry_summary: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Dynamically calculates optimal batch size of raw strings and prunes canonical registry summary
        to maximize throughput while strictly complying with TPM (Tokens Per Minute) and RPM limits.
        """
        if not unresolved_strings:
            return [], []

        # Target max tokens allowed per request (safe cap of 8,000 tokens to stay well under 250k TPM)
        target_tokens_per_req = 8000

        # Generic stopwords to exclude from registry matching to avoid matching every entity
        generic_stopwords = {
            "university", "college", "institute", "school", "department", "center", "centre", 
            "laboratory", "lab", "inc", "ltd", "corp", "corporation", "llc", "group", "faculty", 
            "academy", "technology", "science", "national", "state", "research", "engineering", 
            "system", "systems", "china", "japan", "usa", "germany", "france", "spain", "italy", 
            "korea", "canada", "uk", "dept"
        }

        # Sample unresolved strings for specific entity keywords (ignoring generic stopwords)
        sample = unresolved_strings[:50]
        keywords = set()
        for s in sample:
            for w in re.findall(r"\w+", s.lower()):
                if len(w) >= 4 and w not in generic_stopwords:
                    keywords.add(w)

        # Prune registry summary to at most 100 relevant entries
        pruned_registry = []
        if len(canonical_registry_summary) <= 100:
            pruned_registry = canonical_registry_summary
        else:
            for e in canonical_registry_summary:
                c_name = e.get("canonical_name", "").lower()
                aliases = " ".join(e.get("known_aliases", [])).lower()
                if any(w in c_name or w in aliases for w in keywords):
                    pruned_registry.append(e)
                if len(pruned_registry) >= 100:
                    break

            # Fallback to top 100 entries if keyword matches are sparse
            if len(pruned_registry) < 50:
                seen_ids = {e["canonical_id"] for e in pruned_registry}
                for e in canonical_registry_summary[:100]:
                    if e["canonical_id"] not in seen_ids:
                        pruned_registry.append(e)
                    if len(pruned_registry) >= 100:
                        break

        # Estimate prompt token size (~20 tokens per registry entry + system prompt overhead)
        est_registry_tokens = len(pruned_registry) * 20 + 400
        avail_token_budget = max(2000, target_tokens_per_req - est_registry_tokens)

        # Estimate batch count (~15 tokens per raw string, max 50 strings per batch)
        optimal_count = max(10, min(50, avail_token_budget // 15))
        batch = unresolved_strings[:optimal_count]

        est_total_tokens = est_registry_tokens + len(batch) * 15
        logger.info(
            f"Dynamic Batching: Selected {len(batch)} raw strings with {len(pruned_registry)} canonical registry entries. "
            f"(Est. Tokens: {est_total_tokens} / Target Cap: {target_tokens_per_req} TPM Limit: {self.rate_limiter.tpm_capacity:.0f})"
        )
        return batch, pruned_registry

    def normalize_batch(
        self,
        raw_strings: List[str],
        canonical_registry_summary: List[Dict[str, Any]],
        max_retries: int = 10,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batches raw affiliation strings and queries Gemini API for entity resolution.
        Applies rate limiting and automatic cool-off period with exponential backoff on 429.
        """
        if not raw_strings:
            return {}

        if not self.api_key:
            err_msg = "GEMINI_API_KEY is missing. Cannot perform LLM normalization."
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        prompt_payload = {
            "existing_canonical_organizations": canonical_registry_summary,
            "raw_affiliation_strings": raw_strings,
        }

        request_body = {
            "contents": [
                {
                    "parts": [
                        {"text": SYSTEM_PROMPT},
                        {"text": f"INPUT PAYLOAD:\n{json.dumps(prompt_payload, indent=2)}"},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

        url = f"{GEMINI_API_ENDPOINT}?key={self.api_key}"

        for attempt in range(max_retries):
            self.rate_limiter.acquire()

            try:
                response = self.session.post(url, json=request_body, timeout=120)
                self._inspect_headers(response)

                if response.status_code == 429 or response.status_code == 503:
                    quota_reason = "Unknown"
                    resp_text = response.text if response.text else ""
                    try:
                        err_data = response.json().get("error", {})
                        details = err_data.get("details", [])
                        for d in details:
                            meta = d.get("metadata", {})
                            if "quota_limit" in meta or "quota_metric" in meta:
                                quota_reason = meta.get("quota_limit") or meta.get("quota_metric")
                                break
                        if quota_reason == "Unknown" and err_data.get("message"):
                            quota_reason = err_data.get("message")
                    except Exception:
                        quota_reason = resp_text[:200] if resp_text else "No response body"

                    retry_secs = None
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        try:
                            retry_secs = float(retry_after)
                        except ValueError:
                            pass

                    if retry_secs is None and resp_text:
                        # Extract "Please retry in X.XXXXs" or similar from response text
                        match = re.search(r"Please retry in (\d+(?:\.\d+)?)s", resp_text, re.IGNORECASE)
                        if match:
                            try:
                                retry_secs = float(match.group(1))
                            except ValueError:
                                pass

                    if retry_secs is not None:
                        # Round UP to nearest multiple of 10 seconds (minimum 10s)
                        sleep_time = max(10.0, math.ceil(retry_secs / 10.0) * 10.0)
                        logger.warning(
                            f"Gemini API Rate limited ({response.status_code}). Quota limit exceeded: '{quota_reason}'. "
                            f"API requested retry in {retry_secs:.2f}s -> Cooling off for {sleep_time:.0f}s (Attempt {attempt+1}/{max_retries})"
                        )
                    else:
                        sleep_time = max(10.0, math.ceil(((2 ** attempt) * 5.0) / 10.0) * 10.0)
                        logger.warning(
                            f"Gemini API Rate limited ({response.status_code}). Quota limit exceeded: '{quota_reason}'. "
                            f"Cooling off for {sleep_time:.0f}s (Attempt {attempt+1}/{max_retries})"
                        )

                    time.sleep(sleep_time)
                    continue

                response.raise_for_status()
                data = response.json()

                # Extract text output
                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError(f"Gemini API returned no candidates: {data}")

                text_content = candidates[0]["content"]["parts"][0]["text"]
                result_json = json.loads(text_content)
                raw_res = result_json.get("resolutions", {})

                resolutions: Dict[str, Dict[str, Any]] = {}
                if isinstance(raw_res, dict):
                    resolutions = raw_res
                elif isinstance(raw_res, list):
                    for item in raw_res:
                        if isinstance(item, dict):
                            raw_k = item.get("raw_string") or item.get("raw_affiliation_string") or item.get("raw")
                            if raw_k:
                                resolutions[str(raw_k).strip()] = item

                logger.info(f"Gemini API returned {len(resolutions)} resolutions for batch of {len(raw_strings)} strings")
                return resolutions

            except (requests.exceptions.RequestException, Exception) as e:
                if attempt < max_retries - 1:
                    sleep_time = max(15.0, (2 ** attempt) * 3.0 + random.uniform(1.0, 3.0))
                    logger.warning(
                        f"Error querying Gemini API ({e}). Cooling off for {sleep_time:.2f}s (Attempt {attempt+1}/{max_retries})"
                    )
                    time.sleep(sleep_time)
                    continue
                logger.error(f"Failed Gemini API normalization query after {max_retries} attempts: {e}", exc_info=True)
                raise e

        err_msg = f"Exhausted retries ({max_retries}) querying Gemini API for batch normalization."
        logger.error(err_msg, exc_info=True)
        raise RuntimeError(err_msg)

    def resolve_openalex_source(
        self,
        venue: str,
        candidates: List[Dict[str, Any]],
        max_retries: int = 3,
    ) -> Optional[str]:
        """
        Uses Gemini LLM to analyze candidate OpenAlex source records for a venue,
        and select the single best matching primary source ID (e.g. 'S4363608614').
        """
        if not candidates:
            return None
        if not self.api_key:
            logger.warning("GEMINI_API_KEY missing. Skipping Gemini LLM OpenAlex source selection.")
            return None

        # Prefer candidates with non-zero works_count
        valid_candidates = [c for c in candidates if (c.get("works_count") or 0) > 0]
        if not valid_candidates:
            valid_candidates = candidates

        cand_summaries = []
        for c in valid_candidates:
            cand_summaries.append({
                "id": c.get("id"),
                "display_name": c.get("display_name"),
                "type": c.get("type"),
                "works_count": c.get("works_count"),
                "first_publication_year": c.get("first_publication_year"),
                "last_publication_year": c.get("last_publication_year"),
            })

        system_prompt = (
            "You are an expert academic publication metadata assistant.\n"
            "Given a target academic conference or journal venue string (e.g. 'IROS' or 'ICRA') and a list of candidate OpenAlex source objects, "
            "select the single best matching primary OpenAlex Source ID (e.g. 'S4363608614') that contains the publication works (works_count > 0).\n"
            "Pick the candidate that represents the main conference/journal proceedings series.\n"
            "Output JSON format:\n"
            '{\n  "selected_source_id": "S4363608614",\n  "reasoning": "Explanation..."\n}'
        )

        request_body = {
            "contents": [
                {
                    "parts": [
                        {"text": system_prompt},
                        {"text": f"TARGET VENUE: {venue}\nCANDIDATES:\n{json.dumps(cand_summaries, indent=2)}"},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

        url = f"{GEMINI_API_ENDPOINT}?key={self.api_key}"

        for attempt in range(max_retries):
            self.rate_limiter.acquire()
            try:
                response = self.session.post(url, json=request_body, timeout=30)
                self._inspect_headers(response)
                if response.status_code in (429, 503):
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                data = response.json()
                text_content = data["candidates"][0]["content"]["parts"][0]["text"]
                res_json = json.loads(text_content)
                sel_id = res_json.get("selected_source_id", "")
                if sel_id:
                    sel_id = sel_id.split("/")[-1]
                    logger.info(f"Gemini selected OpenAlex source ID '{sel_id}' for venue '{venue}' (Reasoning: {res_json.get('reasoning')})")
                    return sel_id
            except Exception as e:
                logger.warning(f"Gemini source resolution attempt {attempt+1} failed: {e}")

        return None


