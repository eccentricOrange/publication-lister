# Implementation Plan: Academic Conference & Journal Affiliation Tracker

Build a modular, extensible, CLI-driven Python codebase to extract, deduplicate, normalize, and analyze institutional affiliation statistics across robotics and AI publication venues (starting with ICRA 2017–2026, extensible to IROS, RA-L, TRO, CoRL, CVPR, NeurIPS, ECCV).

## User Review Required

> [!IMPORTANT]
> - **Strict Exception Handling & Logging**:
>   - **No Dummy/Fallback Data**: Extractors and normalizers will **NEVER** use dummy test cases or mock fallback data. If API keys are missing, network requests fail, or parsing errors occur, the system will log the exact error traceback and **immediately raise an exception and halt execution**.
>   - **Centralized Single-Config Logging**: Logging will be configured **exclusively at the entrypoint (`main.py`)**. All submodules acquire their loggers via `logging.getLogger(__name__)`.
>   - **Log Format**: `[%(asctime)s] [PID:%(process)d] [%(name)s] [%(levelname)s] %(message)s`. Output is simultaneously sent to **stdout** and written to `logs/tracker.log`.
> - **Initial API-Driven Rate Limit Query**:
>   - Upon program startup, the client queries the target API to extract the rate limit from initial response headers, then enforces this rate limit pacing throughout the execution run.
> - **Canonical ID Format**: Strictly follows `[TYPE:3]-[ID:5]-[SLUG:6]` (e.g., `UNI-00142-UCSDCA`, `COM-00028-GOOGUS`, `GOV-00009-NASAJPL`).
> - **Rules Enforcement**:
>   - **University vs. Company**: Campuses stay distinct (`UC Los Angeles` vs `UC San Diego`), while corporate subsidiaries collapse to parent entities (`Google LLC`).
>   - **Within-Paper Deduplication**: An institution is counted at most once per paper.
>   - **Paper-Level Multi-Affiliation**: Multiple distinct institutions on a paper each gain 1 count for that paper.

## Detailed Module Specifications

### Data Ingestion & Extractor Details (`src/extractors/`)

1. **`base.py` (`BaseExtractor`)**:
   - Manages raw caching in `data/raw/<venue>/<venue>_<year>.json`.
   - Handles network requests using `requests.Session` with retry logic.
   - Queries API on initial run to inspect rate-limit response headers (`x-ratelimit-limit`, `x-ratelimit-remaining`, etc.) and sets pacing limit for the run.
   - Raises explicit exceptions with full tracebacks on HTTP errors or missing configuration.

2. **`ieee_xplore.py` (`IEEEExtractor`)**:
   - Queries IEEE Xplore REST API endpoint (`https://ieeexploreapi.ieee.org/api/v1/search/articles`).
   - Paginated fetching of paper metadata for target publication title and year.
   - Extracts paper ID, article title, and author affiliation strings.

3. **`scopus.py` (`ScopusExtractor`)**:
   - Interfaces with Elsevier Scopus Search API (`https://api.elsevier.com/content/search/scopus`).
   - Retrieves publication entries for conference proceedings/journals with affiliated institution metadata.

4. **`conference_schedule.py` (`ConferenceScheduleExtractor`)**:
   - Scraper/parser for online conference program websites (e.g., Papercept schedule pages, epapers program indexes).
   - Uses `BeautifulSoup` to parse session HTML tables and paper cards.
   - Extracts paper ID, title, author list, and author institution text blocks attached to each paper.

---

## Proposed Changes

### Packaging & Infrastructure

#### [NEW] [pyproject.toml](file:///mnt/win-shared/edu/graduate-studies/publication-lister/pyproject.toml)
- Packaging metadata with third-party dependencies: `requests`, `python-dotenv`, `google-genai`, `beautifulsoup4`. (Standard library `pathlib` excluded).
- Exposes CLI entrypoint `affiliation-tracker = main:cli`.

#### [NEW] [.env.example](file:///mnt/win-shared/edu/graduate-studies/publication-lister/.env.example)
- Required environment variable templates (`GEMINI_API_KEY`, `IEEE_API_KEY`, `SCOPUS_API_KEY`).

#### [NEW] [.gitignore](file:///mnt/win-shared/edu/graduate-studies/publication-lister/.gitignore)
- Ignores `.env`, `.venv/`, `__pycache__/`, `logs/`, build artifacts, and output data directories except `.gitkeep` / canonical JSON.

---

### Core Source Code Modules (`src/`)

#### [NEW] [src/logger.py](file:///mnt/win-shared/edu/graduate-studies/publication-lister/src/logger.py)
- `setup_logging()` helper called exclusively in `main.py`.
- Formatter string: `[%(asctime)s] [PID:%(process)d] [%(name)s] [%(levelname)s] %(message)s`.
- Sets up `StreamHandler` (stdout) and `FileHandler` (`logs/tracker.log`).

#### [NEW] [src/config.py](file:///mnt/win-shared/edu/graduate-studies/publication-lister/src/config.py)
- Configuration management utilizing `python-dotenv` and standard `pathlib.Path`.

#### [NEW] [src/extractors/base.py](file:///mnt/win-shared/edu/graduate-studies/publication-lister/src/extractors/base.py)
- Abstract base extractor class. Implements caching and initial rate-limit header parsing.

#### [NEW] [src/extractors/ieee_xplore.py](file:///mnt/win-shared/edu/graduate-studies/publication-lister/src/extractors/ieee_xplore.py)
- IEEE Xplore API extractor implementation.

#### [NEW] [src/extractors/scopus.py](file:///mnt/win-shared/edu/graduate-studies/publication-lister/src/extractors/scopus.py)
- Scopus Search API extractor implementation.

#### [NEW] [src/extractors/conference_schedule.py](file:///mnt/win-shared/edu/graduate-studies/publication-lister/src/extractors/conference_schedule.py)
- Conference HTML/JSON schedule parser using `BeautifulSoup`.

#### [NEW] [src/registry/organization_registry.py](file:///mnt/win-shared/edu/graduate-studies/publication-lister/src/registry/organization_registry.py)
- Central registry manager for `data/canonical_organizations.json`.
- Validates and maintains schema: `[TYPE:3]-[ID:5]-[SLUG:6]`.

#### [NEW] [src/normalizer/rate_limiter.py](file:///mnt/win-shared/edu/graduate-studies/publication-lister/src/normalizer/rate_limiter.py)
- Rate limiter initialized via API header limit detection at startup to enforce request pacing.

#### [NEW] [src/normalizer/gemini_client.py](file:///mnt/win-shared/edu/graduate-studies/publication-lister/src/normalizer/gemini_client.py)
- Gemini API interface using `google-genai`. Queries initial API limits and applies backoff on HTTP 429.

#### [NEW] [src/normalizer/affiliation_normalizer.py](file:///mnt/win-shared/edu/graduate-studies/publication-lister/src/normalizer/affiliation_normalizer.py)
- Normalizer orchestrator that maps raw strings via local registry first and Gemini API second. Writes output to `data/normalized/<venue>_<year>_normalized.json`.

#### [NEW] [src/exporters/matrix_exporter.py](file:///mnt/win-shared/edu/graduate-studies/publication-lister/src/exporters/matrix_exporter.py)
- Aggregates yearly paper counts applying within-paper deduplication and multi-affiliation rules. Exports to `data/output/<venue>_affiliations_<start_year>_<end_year>.csv`.

#### [NEW] [main.py](file:///mnt/win-shared/edu/graduate-studies/publication-lister/main.py)
- CLI entrypoint built using `argparse`. Calls `setup_logging()` once at startup.

---

### Data & Documentation

#### [NEW] [data/canonical_organizations.json](file:///mnt/win-shared/edu/graduate-studies/publication-lister/data/canonical_organizations.json)
- Initial canonical organization registry file.

#### [NEW] [README.md](file:///mnt/win-shared/edu/graduate-studies/publication-lister/README.md)
- Complete technical manual, API key setup guide, logging guidelines, and execution examples.

## Verification Plan

### Automated Tests
- Test cases for:
  - Canonical ID format validation (`UNI-00001-STANFD`).
  - Startup API rate-limit header parsing.
  - Centralized logger configuration (verifying `[asctime] [PID:...] [name] [levelname]` format with dual stdout/file logging).
  - Matrix deduplication and export logic.
  - Strict exception propagation (no dummy code, full traceback logging and execution halting).

### Manual Verification
- Run CLI subcommands and inspect `logs/tracker.log` to confirm logging format and error propagation behavior.
