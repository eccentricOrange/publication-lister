# Academic Conference & Journal Affiliation Tracker

A modular, extensible, CLI-driven Python codebase to extract, deduplicate, normalize, and analyze institutional affiliation statistics across robotics and AI publication venues (starting with ICRA 2017–2026, extensible to IROS, RA-L, TRO, CoRL, CVPR, NeurIPS, ECCV).

The system generates a consolidated matrix CSV where each row represents a canonical institution, each column represents a publication year, and cell values represent the number of distinct accepted papers authored by at least one researcher affiliated with that institution in that year.

---

## Core Counting & Hierarchy Rules

1. **Paper-Level Multi-Affiliation Increment:** If a paper has co-authors across multiple distinct institutions (e.g., Stanford and MIT), both institutions gain +1 count for that paper.
2. **Within-Paper De-duplication:** If a paper lists multiple co-authors from the same institution, that institution is counted **only once** for that paper.
3. **Institutional Hierarchy Rules:**
   - **Universities:** Distinct campuses remain distinct entities (e.g., `University of California, Los Angeles` vs. `University of California, San Diego`). Campuses are **not** rolled up into parent systems.
   - **Companies:** Global/regional subsidiaries aggregate into a single parent entity (e.g., Google India, Google Brain, Google Zurich -> `Google LLC`).
   - **Labs/Gov Agencies:** Independent research labs and government agencies remain distinct (e.g., Max Planck Institutes, NASA JPL, CNRS).
4. **Canonical ID Format:** Fixed-length unique identifiers following `[TYPE:3]-[ID:5]-[SLUG:6]` (e.g., `UNI-00142-UCSDCA`, `COM-00028-GOOGUS`, `GOV-00009-NASAJPL`).

---

## Project Structure

```
publication-lister/
├── pyproject.toml                     # Modern Python packaging configuration
├── .env.example                       # API key templates
├── .gitignore                         # Clean git ignore configuration
├── README.md                          # Technical manual & documentation
├── main.py                            # CLI entrypoint (subcommands: extract, normalize, export, pipeline)
├── src/
│   ├── __init__.py
│   ├── config.py                      # Environment & path management (pathlib based)
│   ├── logger.py                      # Centralized logging setup
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseExtractor with caching & initial API limit header parsing
│   │   ├── ieee_xplore.py             # IEEE Xplore REST API extractor
│   │   ├── scopus.py                  # Elsevier Scopus REST API extractor
│   │   └── conference_schedule.py     # BeautifulSoup HTML/JSON program schedule parser
│   ├── registry/
│   │   ├── __init__.py
│   │   └── organization_registry.py   # Central canonical registry manager ([TYPE:3]-[ID:5]-[SLUG:6])
│   ├── normalizer/
│   │   ├── __init__.py
│   │   ├── rate_limiter.py            # Token-Bucket rate limiter initialized from startup API headers
│   │   ├── gemini_client.py           # Gemini API client with exponential backoff on 429
│   │   └── affiliation_normalizer.py  # Local registry search + batched Gemini LLM resolution
│   └── exporters/
│       ├── __init__.py
│       └── matrix_exporter.py         # CSV matrix builder (deduplicated, sorted descending by total)
├── tests/                             # Comprehensive unit test suite
│   ├── test_registry.py
│   ├── test_logger_and_rate_limiter.py
│   └── test_normalizer_and_exporter.py
└── data/
    ├── raw/                           # Raw extraction JSON artifacts (data/raw/<venue>/<venue>_<year>.json)
    ├── normalized/                    # Normalized paper artifacts (data/normalized/<venue>_<year>_normalized.json)
    ├── output/                        # Matrix CSV exports (data/output/<venue>_affiliations_<start>_<end>.csv)
    └── canonical_organizations.json   # Persistent central registry file
```

---

## Setup & API Key Registration

### 1. Prerequisites
- Python >= 3.10

### 2. Installation
Clone the repository and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. API Key Registration & Configuration
Copy `.env.example` to `.env` and fill in your academic API credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Gemini API Key (for LLM normalization)
# Register at: https://aistudio.google.com/
GEMINI_API_KEY=your_gemini_api_key_here

# IEEE Xplore API Key (for IEEE proceedings e.g. ICRA, IROS, RA-L)
# Register at: https://developer.ieee.org/
IEEE_API_KEY=your_ieee_api_key_here

# Elsevier Scopus API Key (for Scopus indexing)
# Register at: https://dev.elsevier.com/
SCOPUS_API_KEY=your_scopus_api_key_here
```

---

## Usage Guide & CLI Examples

The CLI entrypoint is available either via `python main.py` or the installed binary `affiliation-tracker`.

### Subcommand 1: `extract`
Extracts raw paper metadata and raw author affiliation strings into `data/raw/<venue>/<venue>_<year>.json`.

```bash
# Extract ICRA papers 2017 to 2025 via IEEE Xplore API
python main.py extract --venue ICRA --year-start 2017 --year-end 2025 --source ieee

# Extract ICRA 2026 from online conference schedule HTML when publisher indexing is pending
python main.py extract --venue ICRA --year-start 2026 --year-end 2026 --source schedule --schedule-file data/icra2026_program.html
```

### Subcommand 2: `normalize`
Maps raw affiliation strings against `data/canonical_organizations.json` locally, and batches novel strings for resolution via the Gemini API (`gemini-2.5-flash-lite`). Outputs to `data/normalized/<venue>_<year>_normalized.json`.

```bash
python main.py normalize --venue ICRA --year-start 2017 --year-end 2026
```

### Subcommand 3: `export`
Aggregates normalized yearly paper records, applies within-paper deduplication, and exports the sorted CSV matrix to `data/output/<venue>_affiliations_<start_year>_<end_year>.csv`.

```bash
python main.py export --venue ICRA --year-start 2017 --year-end 2026
```

### Subcommand 4: `pipeline`
Runs the full end-to-end extraction, normalization, and export flow:

```bash
python main.py pipeline --venue ICRA --year-start 2017 --year-end 2026 --source all
```

---

## Exception Handling & Dual Logging

- **Logging Format**: Centrally configured at entrypoint with format:
  `[%(asctime)s] [PID:%(process)d] [%(name)s] [%(levelname)s] %(message)s`
- **Output Targets**: Logs are simultaneously written to `stdout` and saved to `logs/tracker.log`.
- **Strict Error Policy**: The codebase **never silently swallows errors or uses mock dummy data**. Any missing API key, network failure, or parsing error will log the exact exception traceback and immediately halt execution.

---

## Running Unit Tests

Run the test suite using Python's standard `unittest`:

```bash
python -m unittest discover -s tests
```

