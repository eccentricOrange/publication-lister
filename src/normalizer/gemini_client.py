import json
import logging
import math
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import errors, types

from src.config import DEFAULT_GEMINI_MODEL, GEMINI_API_KEY
from src.normalizer.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

GEMINI_MODEL = DEFAULT_GEMINI_MODEL

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


def parse_gemini_json(text_content: str) -> Any:
    """
    Parses JSON output from Gemini LLM responses, applying multi-tier cleanup
    for markdown code fences, leading/trailing prose, trailing commas, and unescaped control characters.
    """
    if not text_content or not text_content.strip():
        raise ValueError("Empty response text from Gemini LLM")

    cleaned = text_content.strip()

    # 1. Strip markdown code fences (```json ... ``` or ``` ...)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # 2. Try standard json.loads
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Extract JSON object/array substring if extra prose surrounds it
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        try:
            return json.loads(extracted)
        except (json.JSONDecodeError, ValueError):
            cleaned = extracted

    # 4. Remove trailing commas before } or ]
    cleaned_fix = re.sub(r",\s*([\}\]])", r"\1", cleaned)
    try:
        return json.loads(cleaned_fix)
    except (json.JSONDecodeError, ValueError):
        pass

    # 5. Remove unprintable control characters (except \n, \r, \t)
    cleaned_fix = "".join(ch for ch in cleaned_fix if ord(ch) >= 32 or ch in "\n\r\t")
    try:
        return json.loads(cleaned_fix)
    except (json.JSONDecodeError, ValueError):
        pass

    # Final attempt: re-raise original JSON error
    return json.loads(cleaned)


class GeminiClient:
    """
    Gemini API Client for affiliation string normalization using official google-genai SDK.
    Integrates TokenBucketRateLimiter, pipe-delimited compact registry context, minified payload serialization,
    streamlined resolution outputs, native HttpOptions, and adaptive cool-off handling.
    """

    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = DEFAULT_GEMINI_MODEL,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
    ):
        self.api_key = api_key
        self.model = model or DEFAULT_GEMINI_MODEL
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
            # Use 45-second timeout (45,000 ms) and native HttpRetryOptions with limited attempts for transient errors
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    timeout=45_000,
                    retry_options=types.HttpRetryOptions(
                        attempts=2,
                        initial_delay=2.0,
                        max_delay=10.0,
                        exp_base=2.0,
                        http_status_codes=[503, 500, 502, 429],
                    ),
                ),
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

    def calculate_dynamic_batch(
        self,
        unresolved_strings: List[str],
        canonical_registry_summary: Optional[List[Dict[str, Any]]] = None,
        target_batch_size: int = 50,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Calculates unresolved strings batch for LLM query.
        Prunes registry using pipe-delimited formatting to stay under token limits.
        """
        if not unresolved_strings:
            return [], []

        registry_summary = canonical_registry_summary or []
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
        if len(registry_summary) <= 120:
            pruned_registry = registry_summary
        else:
            for e in registry_summary:
                c_name = e.get("canonical_name", "").lower()
                aliases = " ".join(e.get("known_aliases", [])).lower()
                if any(w in c_name or w in aliases for w in keywords):
                    pruned_registry.append(e)
                if len(pruned_registry) >= 120:
                    break

            if len(pruned_registry) < 50:
                seen_ids = {e["canonical_id"] for e in pruned_registry}
                for e in registry_summary[:120]:
                    if e["canonical_id"] not in seen_ids:
                        pruned_registry.append(e)
                    if len(pruned_registry) >= 120:
                        break

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
        canonical_registry_summary: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 10,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batches raw affiliation strings and queries Gemini API for entity resolution using google-genai SDK.
        Uses minified JSON serialization (separators=(',', ':')) and compact pipe-delimited registry formatting.
        """
        if not raw_strings:
            return {}

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
                    model=self.model,
                    contents=contents,
                    config=config,
                )

                if response.sdk_http_response and hasattr(response.sdk_http_response, "headers"):
                    self._inspect_headers(response.sdk_http_response.headers)

                text_content = self._extract_response_text(response)
                result_json = parse_gemini_json(text_content)
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
                is_503 = (isinstance(e, errors.APIError) and e.code == 503) or "503" in err_str or "UNAVAILABLE" in err_str or "Service Unavailable" in err_str
                is_504 = "504" in err_str or "DEADLINE_EXCEEDED" in err_str or "timed out" in err_str.lower() or "timeout" in err_str.lower()
                is_json_err = isinstance(e, (json.JSONDecodeError, ValueError)) or "JSONDecodeError" in err_str or "Expecting property name" in err_str or "Expecting value" in err_str or r"Invalid \escape" in err_str

                retry_secs = None
                match = re.search(r"Please retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                if match:
                    try:
                        retry_secs = float(match.group(1))
                    except ValueError:
                        pass

                # If request timed out (504) or returned unparseable JSON, split the batch in half!
                if (is_504 or is_json_err) and len(raw_strings) > 1:
                    mid = len(raw_strings) // 2
                    reason = "timed out (504)" if is_504 else "returned malformed JSON"
                    logger.warning(
                        f"Gemini API normalization {reason} on batch of {len(raw_strings)} strings. "
                        f"Splitting batch into sub-batches of {mid} and {len(raw_strings) - mid} strings."
                    )
                    res1 = self.normalize_batch(raw_strings[:mid], canonical_registry_summary=canonical_registry_summary, max_retries=max_retries)
                    res2 = self.normalize_batch(raw_strings[mid:], canonical_registry_summary=canonical_registry_summary, max_retries=max_retries)
                    combined = {}
                    combined.update(res1)
                    combined.update(res2)
                    return combined

                # If a single string produced unparseable JSON, fall back gracefully to a self-resolution
                if is_json_err and len(raw_strings) == 1:
                    single_raw = raw_strings[0]
                    logger.warning(f"Gemini API returned unparseable JSON for single string '{single_raw}'. Resolving as fallback self-entry.")
                    return {single_raw: {"canonical_name": single_raw, "entity_type": "UNI"}}

                if is_429:
                    if retry_secs is not None:
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
                elif is_503:
                    if attempt < max_retries - 1:
                        sleep_time = min(60.0, (2 ** attempt) * 2.0 + random.uniform(1.0, 3.0))
                        logger.warning(
                            f"Gemini API 503 Service Unavailable ({e}). "
                            f"Exponential back-off sleeping for {sleep_time:.2f}s before retry (Attempt {attempt+1}/{max_retries})"
                        )
                        time.sleep(sleep_time)
                        continue
                    else:
                        logger.error(f"Failed Gemini API normalization query after {max_retries} attempts due to 503 Service Unavailable: {e}", exc_info=True)
                        raise e
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

    def resolve_openalex_venue_sources(
        self,
        venue: str,
        candidates: List[Dict[str, Any]],
        short_name: Optional[str] = None,
        search_term: Optional[str] = None,
        max_retries: int = 10,
    ) -> Dict[str, Any]:
        """
        Uses Gemini LLM to analyze candidate OpenAlex source records and sample work DOIs for a venue,
        and select matching primary OpenAlex source IDs, DOI prefixes, and venue frequency (annual vs biennial_even vs biennial_odd).
        Presents both short_name (e.g. 'IROS') and search_term (e.g. 'IEEE/RSJ International Conference...') to LLM.
        """
        if not candidates:
            return {"source_ids": [], "doi_prefixes": [], "frequency": "annual"}
        if not self.api_key:
            logger.warning("GEMINI_API_KEY missing. Skipping Gemini LLM OpenAlex source resolution.")
            return {"source_ids": [], "doi_prefixes": [], "frequency": "annual"}

        cand_summaries = []
        for c in candidates:
            cand_summaries.append({
                "id": c.get("id"),
                "display_name": c.get("display_name"),
                "type": c.get("type"),
                "publisher": c.get("publisher"),
                "host_organization_name": c.get("host_organization_name"),
                "works_count": c.get("works_count"),
                "cited_by_count": c.get("cited_by_count"),
                "first_publication_year": c.get("first_publication_year"),
                "last_publication_year": c.get("last_publication_year"),
                "issn_l": c.get("issn_l"),
                "issn": c.get("issn"),
                "ids": c.get("ids"),
                "sample_doi_prefixes": c.get("sample_doi_prefixes", []),
            })

        system_prompt = (
            "You are an expert academic publication metadata assistant.\n"
            "Given a target academic conference or journal venue short name and search term, and the complete list of candidate OpenAlex source records from the OpenAlex Sources API (including publication counts, publisher/host organization, DOIs, and sample DOI prefixes extracted from real works),\n"
            "analyze all candidates and select:\n"
            "1. ALL matching primary OpenAlex Source IDs (e.g. ['S4363608614']) and valid DOI prefixes (e.g. ['10.1109/icra']) for this venue.\n"
            "2. Venue publication frequency ('annual', 'biennial_even' for conferences held in even years like ECCV, 'biennial_odd' for conferences held in odd years like ICCV, or 'irregular').\n"
            "If multiple source IDs represent different volumes, proceedings series, or years for the same venue, include all relevant source IDs.\n"
            "Output JSON format:\n"
            '{\n  "selected_source_ids": ["S4363608614"],\n  "doi_prefixes": ["10.1109/iros"],\n  "frequency": "biennial_even",\n  "reasoning": "Explanation..."\n}'
        )

        eff_short = short_name or venue
        if isinstance(search_term, list):
            eff_search = ", ".join(f"'{s}'" for s in search_term)
        else:
            eff_search = str(search_term or venue)

        target_info = f"TARGET VENUE SHORT NAME: {eff_short}\nTARGET VENUE SEARCH TERMS: {eff_search}"

        contents = [
            system_prompt,
            f"{target_info}\nCANDIDATES:\n{json.dumps(cand_summaries, separators=(',', ':'), ensure_ascii=False)}",
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
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                if response.sdk_http_response and hasattr(response.sdk_http_response, "headers"):
                    self._inspect_headers(response.sdk_http_response.headers)

                text_content = self._extract_response_text(response)
                res_json = parse_gemini_json(text_content)
                
                raw_ids = res_json.get("selected_source_ids") or res_json.get("selected_source_id") or []
                if isinstance(raw_ids, str):
                    raw_ids = [raw_ids]
                source_ids = [str(x).split("/")[-1] for x in raw_ids if str(x).strip()]

                raw_prefs = res_json.get("doi_prefixes") or res_json.get("doi_prefix") or []
                if isinstance(raw_prefs, str):
                    raw_prefs = [raw_prefs]
                doi_prefixes = [str(x).strip().lower() for x in raw_prefs if str(x).strip()]

                freq = str(res_json.get("frequency", "annual")).strip().lower()
                if freq not in ["annual", "biennial_even", "biennial_odd", "irregular"]:
                    freq = "annual"

                logger.info(
                    f"Gemini resolved source info for venue '{venue}': "
                    f"source_ids={source_ids}, doi_prefixes={doi_prefixes}, frequency={freq} (Reasoning: {res_json.get('reasoning')})"
                )
                return {"source_ids": source_ids, "doi_prefixes": doi_prefixes, "frequency": freq}
            except Exception as e:
                err_str = str(e)
                is_503 = (isinstance(e, errors.APIError) and e.code == 503) or "503" in err_str or "UNAVAILABLE" in err_str or "Service Unavailable" in err_str
                sleep_time = min(60.0, (2 ** attempt) * 2.0 + random.uniform(1.0, 3.0))
                if is_503:
                    logger.warning(
                        f"Gemini source resolution API 503 Service Unavailable ({e}). "
                        f"Exponential back-off sleeping for {sleep_time:.2f}s before retry (Attempt {attempt+1}/{max_retries})"
                    )
                else:
                    logger.warning(
                        f"Gemini source resolution attempt {attempt+1}/{max_retries} failed ({e}). "
                        f"Exponential back-off sleeping for {sleep_time:.2f}s"
                    )
                if attempt < max_retries - 1:
                    time.sleep(sleep_time)

        return {"source_ids": [], "doi_prefixes": [], "frequency": "annual"}

    def resolve_openalex_source(
        self,
        venue: str,
        candidates: List[Dict[str, Any]],
        short_name: Optional[str] = None,
        search_term: Optional[str] = None,
        max_retries: int = 10,
    ) -> Optional[str]:
        """
        Backward compatible wrapper around resolve_openalex_venue_sources.
        Returns the first resolved source ID string or None.
        """
        res = self.resolve_openalex_venue_sources(
            venue, candidates, short_name=short_name, search_term=search_term, max_retries=max_retries
        )
        s_ids = res.get("source_ids", [])
        return s_ids[0] if s_ids else None

