import json
import logging
import math
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import errors, types

from src.config import GEMINI_API_KEY
from src.normalizer.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-flash-lite"

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

REGISTRY FORMAT:
Provided existing canonical organizations are listed in pipe-delimited format:
CANONICAL_ID|CANONICAL_NAME|ENTITY_TYPE|KNOWN_ALIASES

INSTRUCTIONS:
For each raw affiliation string in the input list:
- If it matches an existing canonical organization, map it directly to its 'canonical_id' string.
- If NO existing organization fits, map it to a new object with 'canonical_name' and 'entity_type' (one of ['UNI', 'COM', 'LAB', 'GOV']).

OUTPUT FORMAT:
Return ONLY a valid JSON object mapping raw strings to resolutions:
{
  "resolutions": {
    "<raw_string_1>": "EXISTING_CANONICAL_ID",
    "<raw_string_2>": {
      "canonical_name": "Official Organization Name",
      "entity_type": "UNI"
    }
  }
}
"""


class GeminiClient:
    """
    Gemini API Client for affiliation string normalization using official google-genai SDK.
    Integrates TokenBucketRateLimiter, pipe-delimited compact registry context, minified payload serialization,
    streamlined resolution outputs, native HttpOptions, and adaptive cool-off handling.
    """

    def __init__(self, api_key: str = GEMINI_API_KEY, rate_limiter: Optional[TokenBucketRateLimiter] = None):
        self.api_key = api_key
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter(requests_per_minute=6.0, tokens_per_minute=250000.0)
        self.initial_rate_queried = False
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        """Lazily initializes and returns the google-genai Client."""
        if self._client is None:
            if not self.api_key:
                err_msg = "GEMINI_API_KEY is missing. Cannot initialize Gemini Client."
                logger.error(err_msg, exc_info=True)
                raise ValueError(err_msg)
            # Use 120-second timeout (120,000 ms) via native HttpOptions
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=120_000),
            )
        return self._client

    def _inspect_headers(self, headers: Any) -> None:
        """Inspects rate limit headers for RPM and TPM limits on initial query."""
        if self.initial_rate_queried or not headers:
            return

        rpm_val = None
        tpm_val = None

        h_dict = {}
        try:
            if hasattr(headers, "items"):
                for k, v in headers.items():
                    h_dict[str(k).lower()] = str(v)
        except Exception:
            pass

        for key in ["x-ratelimit-limit-requests", "x-ratelimit-limit-rpm", "x-ratelimit-limit"]:
            if key in h_dict:
                try:
                    rpm_val = float(h_dict[key])
                    break
                except ValueError:
                    pass

        for key in ["x-ratelimit-limit-tokens", "x-ratelimit-limit-tpm"]:
            if key in h_dict:
                try:
                    tpm_val = float(h_dict[key])
                    break
                except ValueError:
                    pass

        if rpm_val or tpm_val:
            self.rate_limiter.update_limits(requests_per_minute=rpm_val, tokens_per_minute=tpm_val)
            self.initial_rate_queried = True
            logger.info(f"Discovered Gemini API limits from response headers: RPM={self.rate_limiter.capacity:.0f}, TPM={self.rate_limiter.tpm_capacity:.0f}")

    def _extract_response_text(self, response: Any) -> str:
        """Extracts text output from GenerateContentResponse or sdk_http_response.body."""
        if hasattr(response, "text") and response.text:
            return response.text

        if hasattr(response, "sdk_http_response") and response.sdk_http_response:
            body_raw = getattr(response.sdk_http_response, "body", None)
            if body_raw:
                try:
                    body_json = json.loads(body_raw)
                    candidates = body_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for p in parts:
                            if isinstance(p, dict) and p.get("text"):
                                return p["text"]
                except Exception:
                    pass

        raise ValueError(f"Gemini API returned empty text response: {response}")

    def _format_registry_pipe_delimited(self, canonical_registry_summary: List[Dict[str, Any]]) -> List[str]:
        """Formats canonical registry summary entries into compact pipe-delimited lines (ID|Name|Type|Aliases)."""
        lines = []
        for e in canonical_registry_summary:
            c_id = e.get("canonical_id", "")
            c_name = e.get("canonical_name", "")
            e_type = e.get("entity_type", "UNI")
            aliases = ",".join(e.get("known_aliases", []))
            lines.append(f"{c_id}|{c_name}|{e_type}|{aliases}")
        return lines

    def create_cached_context(
        self,
        canonical_registry_summary: List[Dict[str, Any]],
        ttl_minutes: int = 60,
    ) -> str:
        """
        Creates a server-side cachedContent resource storing the full canonical registry context.
        Returns the cache resource name (e.g. 'cachedContents/123456789').
        """
        pipe_lines = self._format_registry_pipe_delimited(canonical_registry_summary)
        registry_text = "\n".join(pipe_lines)
        
        cache_config = types.CreateCachedContentConfig(
            system_instruction=SYSTEM_PROMPT,
            contents=[f"CANONICAL ORGANIZATIONS REGISTRY:\n{registry_text}"],
            ttl=f"{ttl_minutes * 60}s",
            display_name="canonical_org_registry",
        )
        
        logger.info(f"Creating Gemini server-side context cache for {len(canonical_registry_summary)} canonical entities (TTL: {ttl_minutes}m)...")
        cache = self.client.caches.create(
            model=GEMINI_MODEL,
            config=cache_config,
        )
        logger.info(f"Successfully created Gemini server-side context cache: {cache.name}")
        return cache.name

    def calculate_dynamic_batch(
        self,
        unresolved_strings: List[str],
        canonical_registry_summary: Optional[List[Dict[str, Any]]] = None,
        target_batch_size: int = 50,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Calculates unresolved strings batch for LLM query.
        When inline context is used, prunes registry using pipe-delimited formatting to stay under token limits.
        When server-side cached context is used, returns up to target_batch_size without inline registry.
        """
        if not unresolved_strings:
            return [], []

        if not canonical_registry_summary:
            batch = unresolved_strings[:max(50, target_batch_size)]
            logger.info(f"Selected batch of {len(batch)} unresolved affiliation strings for Gemini LLM query (Cached Context active).")
            return batch, []

        target_tokens_per_req = 8000
        generic_stopwords = {
            "university", "college", "institute", "school", "department", "center", "centre", 
            "laboratory", "lab", "inc", "ltd", "corp", "corporation", "llc", "group", "faculty", 
            "academy", "technology", "science", "national", "state", "research", "engineering", 
            "system", "systems", "china", "japan", "usa", "germany", "france", "spain", "italy", 
            "korea", "canada", "uk", "dept"
        }

        sample = unresolved_strings[:50]
        keywords = set()
        for s in sample:
            for w in re.findall(r"\w+", s.lower()):
                if len(w) >= 4 and w not in generic_stopwords:
                    keywords.add(w)

        pruned_registry = []
        if len(canonical_registry_summary) <= 120:
            pruned_registry = canonical_registry_summary
        else:
            for e in canonical_registry_summary:
                c_name = e.get("canonical_name", "").lower()
                aliases = " ".join(e.get("known_aliases", [])).lower()
                if any(w in c_name or w in aliases for w in keywords):
                    pruned_registry.append(e)
                if len(pruned_registry) >= 120:
                    break

            if len(pruned_registry) < 50:
                seen_ids = {e["canonical_id"] for e in pruned_registry}
                for e in canonical_registry_summary[:120]:
                    if e["canonical_id"] not in seen_ids:
                        pruned_registry.append(e)
                    if len(pruned_registry) >= 120:
                        break

        # Pipe-delimited line size is ~15 tokens per entry vs 50 tokens for JSON
        est_registry_tokens = len(pruned_registry) * 15 + 300
        avail_token_budget = max(2000, target_tokens_per_req - est_registry_tokens)
        optimal_count = max(10, min(50, avail_token_budget // 15))
        batch = unresolved_strings[:optimal_count]

        est_total_tokens = est_registry_tokens + len(batch) * 15
        logger.info(
            f"Dynamic Batching (Pipe Inline Context): Selected {len(batch)} raw strings with {len(pruned_registry)} compact pipe-delimited registry entries. "
            f"(Est. Tokens: {est_total_tokens} / Target Cap: {target_tokens_per_req} TPM Limit: {self.rate_limiter.tpm_capacity:.0f})"
        )
        return batch, pruned_registry

    def normalize_batch(
        self,
        raw_strings: List[str],
        cached_content: Optional[str] = None,
        canonical_registry_summary: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 10,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batches raw affiliation strings and queries Gemini API for entity resolution using google-genai SDK.
        Uses minified JSON serialization (separators=(',', ':')) and compact pipe-delimited registry formatting.
        """
        if not raw_strings:
            return {}

        if cached_content:
            prompt_payload = {"raw_affiliation_strings": raw_strings}
            contents = f"INPUT PAYLOAD:\n{json.dumps(prompt_payload, separators=(',', ':'), ensure_ascii=False)}"
            config = types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                should_return_http_response=True,
                cached_content=cached_content,
            )
        else:
            pipe_registry = self._format_registry_pipe_delimited(canonical_registry_summary or [])
            prompt_payload = {
                "existing_canonical_organizations": pipe_registry,
                "raw_affiliation_strings": raw_strings,
            }
            contents = [
                SYSTEM_PROMPT,
                f"INPUT PAYLOAD:\n{json.dumps(prompt_payload, separators=(',', ':'), ensure_ascii=False)}",
            ]
            config = types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                should_return_http_response=True,
            )

        for attempt in range(max_retries):
            self.rate_limiter.acquire()

            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=config,
                )

                if response.sdk_http_response and hasattr(response.sdk_http_response, "headers"):
                    self._inspect_headers(response.sdk_http_response.headers)

                text_content = self._extract_response_text(response)
                result_json = json.loads(text_content)
                raw_res = result_json.get("resolutions", {})

                resolutions: Dict[str, Dict[str, Any]] = {}
                if isinstance(raw_res, dict):
                    for k, v in raw_res.items():
                        clean_k = str(k).strip()
                        if isinstance(v, str):
                            resolutions[clean_k] = {"canonical_id": v.strip()}
                        elif isinstance(v, dict):
                            resolutions[clean_k] = v
                elif isinstance(raw_res, list):
                    for item in raw_res:
                        if isinstance(item, dict):
                            raw_k = item.get("raw_string") or item.get("raw_affiliation_string") or item.get("raw")
                            if raw_k:
                                resolutions[str(raw_k).strip()] = item

                logger.info(f"Gemini API returned {len(resolutions)} resolutions for batch of {len(raw_strings)} strings")
                return resolutions

            except (errors.APIError, Exception) as e:
                err_str = str(e)
                is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str
                
                retry_secs = None
                match = re.search(r"Please retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                if match:
                    try:
                        retry_secs = float(match.group(1))
                    except ValueError:
                        pass

                if is_429:
                    if retry_secs is not None:
                        # Round UP to nearest multiple of 10 seconds (minimum 10s)
                        sleep_time = max(10.0, math.ceil(retry_secs / 10.0) * 10.0)
                        logger.warning(
                            f"Gemini API Rate limited (429). API requested retry in {retry_secs:.2f}s -> "
                            f"Cooling off for {sleep_time:.0f}s (Attempt {attempt+1}/{max_retries})"
                        )
                    else:
                        sleep_time = max(10.0, math.ceil(((2 ** attempt) * 5.0) / 10.0) * 10.0)
                        logger.warning(
                            f"Gemini API Rate limited (429). Cooling off for {sleep_time:.0f}s (Attempt {attempt+1}/{max_retries})"
                        )
                    time.sleep(sleep_time)
                    continue
                else:
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

        contents = [
            system_prompt,
            f"TARGET VENUE: {venue}\nCANDIDATES:\n{json.dumps(cand_summaries, separators=(',', ':'), ensure_ascii=False)}",
        ]

        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            should_return_http_response=True,
        )

        for attempt in range(max_retries):
            self.rate_limiter.acquire()
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=config,
                )
                if response.sdk_http_response and hasattr(response.sdk_http_response, "headers"):
                    self._inspect_headers(response.sdk_http_response.headers)

                text_content = self._extract_response_text(response)
                res_json = json.loads(text_content)
                sel_id = res_json.get("selected_source_id", "")
                if sel_id:
                    sel_id = sel_id.split("/")[-1]
                    logger.info(f"Gemini selected OpenAlex source ID '{sel_id}' for venue '{venue}' (Reasoning: {res_json.get('reasoning')})")
                    return sel_id
            except Exception as e:
                logger.warning(f"Gemini source resolution attempt {attempt+1} failed: {e}")

        return None
