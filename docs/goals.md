### Unified Prompt for Antigravity IDE

# Project Specification: Academic Conference & Journal Affiliation Tracker

Build a modular, extensible, CLI-driven Python codebase to extract, deduplicate, normalize, and analyze institutional affiliation statistics across robotics and AI publication venues (starting with ICRA 2017–2026, extensible to IROS, RA-L, TRO, CoRL, CVPR, NeurIPS, ECCV). 

The goal is to produce a consolidated matrix CSV where:
- Each row represents a unique institution (university campus, company, research lab, or government agency).
- Each column represents a publication year (e.g., 2017 to 2026).
- Cell values represent the number of distinct accepted papers authored by at least one researcher affiliated with that institution in that year.

---

### Core Data & Counting Rules
1. **Paper-Level Multi-Affiliation Increment:** If a paper has authors across multiple distinct institutions (e.g., Stanford and MIT), increment the count for both institutions by 1 for that year.
2. **Within-Paper De-duplication:** If a single paper lists multiple co-authors from the same institution, count that institution only **once** for that paper.
3. **University vs. Company Hierarchy:**
   - **Universities:** Preserve distinct campuses as distinct entities (e.g., `University of California, Los Angeles` vs. `University of California, San Diego`). Do not roll them up into a single parent system.
   - **Companies:** Aggregate all global/regional subsidiaries into a single parent entity (e.g., Google India, Google Brain, Google Zurich -> `Google LLC`).
   - **Labs/Gov Agencies:** Keep distinct parent entities or independent research labs distinct (e.g., Max Planck Institutes, NASA JPL, CNRS).
4. **Naming Convention:** Maintain the formal, official institutional name across all outputs.

---

### Architecture & System Design

#### 1. Data Ingestion & Extraction Modules (`src/extractors/`)
- Design an extensible base extractor class (`BaseExtractor`) with common caching, batching, and network retry logic.
- Implement source-specific submodules:
  - `ieee_xplore.py`: Interface with the official IEEE Xplore API using standard institutional API access (complying strictly with institutional and API terms of service).
  - `scopus.py`: Interface with Elsevier Scopus API (via institutional key / `pybliometrics` or direct REST).
  - `conference_schedule.py`: Pluggable parser for official online conference programs/schedules when official publisher indexes are delayed (e.g., ICRA 2026).
- **Intermediate Artifact:** Extractor output must dump raw, un-deduplicated counts into standardized JSON files under `data/raw/<venue>/<venue>_<year>.json` containing raw paper IDs, raw author affiliation strings, and raw affiliation frequencies.

#### 2. Canonical Organization Registry (`src/registry/`)
- Maintain a persistent central registry (`data/canonical_organizations.json`).
- Each organization must have a fixed-length unique identifier following this strict schema:
  `[TYPE:3]-[ID:5]-[SLUG:6]` (e.g., `UNI-00142-UCSDCA`, `COM-00028-GOOGUS`, `GOV-00009-NASAJPL`).
  - First segment: Entity type (`UNI`, `COM`, `LAB`, `GOV`).
  - Second segment: Zero-padded sequential integer ID (`00001`, `00002`, ...).
  - Third segment: Deterministic uppercase short code / slug.
- Registry fields per entry: `canonical_id`, `canonical_name`, `entity_type`, `known_aliases` (list of strings).

#### 3. LLM-Based Affiliation Resolution (`src/normalizer/`)
- When processing raw affiliation strings, look up existing entries against `canonical_organizations.json` first.
- For unresolved strings or cluster resolution, batch unfamiliar affiliation strings and query the Gemini API (`gemini-3.1-flash-lite` or current Flash-Lite tier).
- **Prompting & Payload:**
  - Send the existing canonical registry alongside the batch of raw author strings.
  - Instruct the model to:
    1. Match each raw string to an existing canonical ID if it refers to the same entity (respecting the university campus vs. corporate rollup rules).
    2. If no match exists, propose a new canonical entry, correct entity type, official formal name, and slug.
- **Dynamic Rate Limiting & Throttling:**
  - Do not use brittle hardcoded delays. Implement an adaptive client utilizing:
    1. Dynamic inspection of rate-limiting response headers (e.g., `x-ratelimit-*`, retry-after) when available.
    2. A Token-Bucket / Leaky-Bucket rate limiter configured via environment variables with a conservative default (e.g., 10–15 RPM for free/low tiers).
    3. Exponential backoff with jitter on HTTP 429 (`RESOURCE_EXHAUSTED`).
- Save normalized paper-affiliation mappings to `data/normalized/<venue>_<year>_normalized.json`.

#### 4. Matrix Aggregation & Export (`src/exporters/`)
- Aggregate normalized yearly counts per canonical organization.
- Export to a clean CSV (`data/output/<venue>_affiliations_<start_year>_<end_year>.csv`):
  - Columns: `canonical_id`, `canonical_name`, `entity_type`, `2017`, `2018`, ..., `2026`, `total`.
  - Sort descending by `total`.

---

### Project Structure & Standards
Follow modern Python packaging standards:
- CLI entrypoint built using `argparse` with subcommands: `extract`, `normalize`, `export`, and `pipeline` (runs full end-to-end flow).
- Use `pathlib.Path` across the entire codebase; no raw string file manipulation.
- Environment variables managed via `python-dotenv` reading from `.env` (e.g., `GEMINI_API_KEY`, `IEEE_API_KEY`, `SCOPUS_API_KEY`, `GEMINI_RPM_LIMIT`).
- Configuration via `pyproject.toml` (compatible with `uv` or `pip`).
- Clean `.gitignore` (ignoring `.env`, virtual environments, `__pycache__`, raw cache files, and build artifacts).

Provide the complete file and folder hierarchy, fully implemented Python source files, and a comprehensive `README.md` with execution examples and instructions on registering academic API keys.
