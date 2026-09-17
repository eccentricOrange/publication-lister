import json
import logging
import random
import time
from typing import Any, Dict, List, Optional
import requests

from src.config import GEMINI_API_KEY
from src.normalizer.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash-lite"
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
    Integrates TokenBucketRateLimiter and exponential backoff on 429.
    Strictly raises exceptions on missing API key or network failures.
    """

    def __init__(self, api_key: str = GEMINI_API_KEY, rate_limiter: Optional[TokenBucketRateLimiter] = None):
        self.api_key = api_key
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter(requests_per_minute=15.0)
        self.session = requests.Session()
        self.initial_rate_queried = False

    def _inspect_headers(self, response: requests.Response) -> None:
        """Inspects rate limit headers on initial query."""
        if self.initial_rate_queried:
            return
        headers = response.headers
        for key in ["x-ratelimit-limit", "x-ratelimit-limit-requests"]:
            if key in headers:
                try:
                    limit_val = float(headers[key])
                    self.rate_limiter.update_limit(limit_val)
                    break
                except ValueError:
                    pass
        self.initial_rate_queried = True

    def normalize_batch(
        self,
        raw_strings: List[str],
        canonical_registry_summary: List[Dict[str, Any]],
        max_retries: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batches raw affiliation strings and queries Gemini API for entity resolution.
        Applies rate limiting and exponential backoff with jitter on 429.
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
                response = self.session.post(url, json=request_body, timeout=60)
                self._inspect_headers(response)

                if response.status_code == 429 or response.status_code == 503:
                    retry_after = response.headers.get("retry-after")
                    if retry_after and retry_after.isdigit():
                        sleep_time = float(retry_after)
                    else:
                        sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                    logger.warning(f"Gemini API Rate limited ({response.status_code}). Retrying in {sleep_time:.2f}s (Attempt {attempt+1}/{max_retries})")
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
                resolutions = result_json.get("resolutions", {})
                return resolutions

            except requests.HTTPError as http_err:
                if response.status_code in (429, 503) and attempt < max_retries - 1:
                    continue
                logger.error(f"HTTP Error querying Gemini API: {http_err}", exc_info=True)
                raise http_err
            except Exception as e:
                logger.error(f"Failed Gemini API normalization query: {e}", exc_info=True)
                raise e

        err_msg = f"Exhausted retries ({max_retries}) querying Gemini API for batch normalization."
        logger.error(err_msg, exc_info=True)
        raise RuntimeError(err_msg)

