# Academic Conference & Journal Affiliation Tracker (`publication-lister`)

A modular, extensible Python toolkit and CLI application to extract, deduplicate, normalize, and analyze institutional affiliation statistics across robotics and AI publication venues (including ICRA, IROS, RA-L, T-RO, CoRL, CVPR, NeurIPS, and ECCV).

`publication-lister` generates consolidated CSV matrix reports where each row represents a canonical institution (e.g. Stanford University, Google LLC), each column represents a publication year, and cell values represent the number of distinct accepted papers authored by at least one researcher affiliated with that institution in that year.

---

## Key Concepts

1. **Paper-Level Multi-Affiliation Increment**:
   - If a paper has co-authors across multiple distinct institutions (e.g., Stanford University and MIT), both institutions receive +1 count for that paper.
2. **Within-Paper De-duplication**:
   - If a paper lists multiple co-authors from the same institution, that institution is counted **only once** for that paper.
3. **Institutional Hierarchy Rules**:
   - **Universities**: Distinct campuses remain distinct entities (e.g., `University of California, Los Angeles` vs. `University of California, San Diego`). Campuses are **not** rolled up into parent university systems.
   - **Companies**: Global, regional, or departmental subsidiaries aggregate into a single parent entity (e.g., Google India, Google Brain, Google Zurich $\rightarrow$ `Google LLC`; FAIR $\rightarrow$ `Meta Platforms, Inc.`).
   - **Labs & Government Agencies**: Independent research institutes and national labs remain distinct entities (e.g., Max Planck Institutes, NASA Jet Propulsion Laboratory, CNRS).
4. **Canonical ID Format**:
   - Fixed-length unique identifiers following `[TYPE:3]-[ID:5]-[SLUG:6]` (e.g., `UNI-00142-UCSDCA`, `COM-00028-GOOGUS`, `GOV-00009-NASAJPL`).
5. **Robust Venue Resolution & Filter Safety**:
   - Converts venue names $\rightarrow$ official OpenAlex Source IDs via the OpenAlex Sources API dis-ambiguated by Gemini LLM (`gemini-3.1-flash-lite`).
   - Dynamically derives DOI prefixes (`10.1109/{acronym}`) to recover papers from untagged OpenAlex proceedings years.
   - **Zero Garbage Fallback**: Strictly avoids unconstrained raw text search (`search: venue`), preventing non-robotics papers (e.g. chemistry or biology papers mentioning "ICRA") from corrupting dataset statistics.
6. **3-Phase Bulk Execution Engine**:
   - **Phase 1 (Bulk Extraction)**: Ingests raw data across all configured venues/years with cursor checkpointing and pause/resume support.
   - **Phase 2 (Global Pooled Normalization)**: Pools all raw affiliation strings across **all** datasets into a single resolution pass against the local registry and Gemini LLM. This maximizes string overlap and minimizes LLM API consumption.
   - **Phase 3 (Matrix Export)**: Exports clean CSV matrix reports sorted descending by total publication volume.

---

## Setup & Prerequisites

### 1. Requirements
- Python $\ge$ 3.10
- API keys for OpenAlex and Google Gemini LLM (plus optional IEEE Xplore / Scopus keys).

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/eccentricOrange/publication-lister.git
cd publication-lister
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. API Key Configuration

Copy `.env.example` to `.env` in the project root directory:

```bash
cp .env.example .env
```

Edit `.env` to insert your credentials:

```env
# OpenAlex API Key (or register mailto for polite pool access)
# Register at: https://openalex.org/
OPENALEX_API_KEY=your_openalex_api_key_here
OPENALEX_MAILTO=your_email@example.com

# Gemini API Key (required for entity normalization & source dis-ambiguation)
# Register at: https://aistudio.google.com/
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: IEEE Xplore & Elsevier Scopus API Keys
IEEE_API_KEY=your_ieee_api_key_here
SCOPUS_API_KEY=your_scopus_api_key_here
```

---

## Configuration Guide (`batch.yaml`)

The bulk execution pipeline is controlled via a YAML configuration file. By default, `publication-lister` looks for a file named `batch.yaml` in the root workspace directory. An example file `batch.example.yaml` is provided in the root directory.

### Example `batch.yaml`

```yaml
# Target venues to harvest and analyze
# Venues can be strings OR objects with search_term and short_name
venues:
  - search_term: "IEEE/RSJ International Conference on Intelligent Robots and Systems"
    short_name: IROS
  - search_term: "International Conference on Robotics and Automation"
    short_name: ICRA
  - search_term: "IEEE/CVF Conference on Computer Vision and Pattern Recognition"
    short_name: CVPR
  - NEURIPS

# Publication years (range or explicit list)
years:
  start: 2020
  end: 2025

# Primary extraction data source (openalex, ieee, scopus, schedule, all)
source: openalex

# Optional per-venue or per-year exceptions/overrides
exceptions:
  - venue: IROS
    years: [2025]
    openalex_source_id: "S4363608614"
```

### YAML Parameter Reference

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :---: | :---: | :--- |
| `venues` | List of Strings / Objects | **Yes** | — | List of target venues. Items can be strings or objects with `search_term` and optional `short_name` (see venue object fields below). |
| `years` | Mapping / List | **Yes** | — | Target publication years. Can be specified as a range (`start: 2017`, `end: 2026`) or as an explicit integer list (`[2021, 2022, 2023]`). |
| `source` | String | No | `openalex` | Primary extraction data source. Supported values: `openalex`, `ieee`, `scopus`, `schedule`, `all`. |
| `exceptions` | List of Objects | No | `[]` | List of venue-specific or year-specific override blocks (see below). |

#### Venue Object Parameters

| Field | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `search_term` | String | **Yes** | Full query term passed to OpenAlex Sources API or IEEE Xplore (e.g. `"IEEE/RSJ International Conference on Intelligent Robots and Systems"`). |
| `short_name` | String | No | Clean short code used for folder/filename paths (e.g. `"IROS"`). If omitted, derived automatically from `search_term`. |
| `openalex_source_id` | String | No | Optional explicit OpenAlex Source ID for this venue. |
| `doi_prefix` | String | No | Optional explicit publisher DOI prefix for this venue. |

#### Exception Block Parameters

| Field | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `venue` | String | **Yes** | The venue short code or name to match against (case-insensitive). |
| `years` | List of Integers | No | Specific years to apply this override to. If omitted, applies to all years for this venue. |
| `openalex_source_id` | String | No | Explicit OpenAlex Source ID (e.g. `"S4363608614"`). Bypasses OpenAlex Sources API resolution. |
| `search_term` | String | No | Custom search query passed to OpenAlex Sources API or IEEE Xplore search. |
| `doi_prefix` | String | No | Explicit publisher DOI prefix (e.g. `"10.1109/lra"` or `"10.1016"`). Bypasses dynamic DOI derivation. |

---

## Usage Guide & CLI Interface

The CLI entrypoint can be run via `python3 main.py` or the `publication-lister` executable.

### 1. Bulk Execution (Recommended)

Run the end-to-end 3-phase bulk pipeline using `batch.yaml`:

```bash
# Uses default batch.yaml in the project root
python3 main.py batch

# Use a custom YAML configuration file
python3 main.py batch --config path/to/my_experiment.yaml

# Enable verbose debug logging
python3 main.py batch --verbose
```

### 2. Single-Venue CLI Subcommands

For targeted single-venue workflows, use individual subcommands:

```bash
# Extract raw metadata for ICRA (2017-2026) via OpenAlex
python3 main.py extract --venue ICRA --year-start 2017 --year-end 2026 --source openalex

# Normalize extracted raw affiliations using local registry + Gemini LLM
python3 main.py normalize --venue ICRA --year-start 2017 --year-end 2026

# Export consolidated CSV matrix
python3 main.py export --venue ICRA --year-start 2017 --year-end 2026

# Full single-venue end-to-end pipeline
python3 main.py pipeline --venue ICRA --year-start 2017 --year-end 2026 --source openalex
```

---

## Understanding the File Structure

| File / Path | Modifiable? | Description |
| :--- | :---: | :--- |
| [main.py](main.py) | No | Main CLI entrypoint; configures argument parsing and dispatches subcommands (`batch`, `extract`, `normalize`, `export`, `pipeline`). |
| [batch.example.yaml](batch.example.yaml) | Reference | Example YAML configuration template for bulk execution. |
| `batch.yaml` | **Yes** | Root configuration file created by the user for defining target venues, year ranges, and exceptions. |
| [src/config.py](src/config.py) | No | Environment variable loading (`.env`) and directory path constants using `pathlib`. |
| [src/logger.py](src/logger.py) | No | Centralized logging configuration routing messages to `stdout` and `logs/tracker.log`. |
| [src/runner/batch_config.py](src/runner/batch_config.py) | No | Dataclass parser for validating `batch.yaml` structure and computing venue/year overrides. |
| [src/runner/bulk_runner.py](src/runner/bulk_runner.py) | No | 3-Phase Bulk Runner Engine coordinating bulk extraction, global pooled normalization, and export. |
| [src/extractors/base.py](src/extractors/base.py) | No | Abstract base extractor handling raw JSON disk caching and cursor token checkpointing. |
| [src/extractors/openalex.py](src/extractors/openalex.py) | No | OpenAlex REST API extractor featuring mandatory Gemini source resolution and DOI prefix fallback. |
| [src/extractors/ieee_xplore.py](src/extractors/ieee_xplore.py) | No | IEEE Xplore REST API extractor for harvesting IEEE conference/journal metadata. |
| [src/extractors/scopus.py](src/extractors/scopus.py) | No | Elsevier Scopus API extractor. |
| [src/extractors/conference_schedule.py](src/extractors/conference_schedule.py) | No | HTML program schedule extractor for unindexed upcoming proceedings. |
| [src/registry/organization_registry.py](src/registry/organization_registry.py) | No | Manager for `data/canonical_organizations.json`. Implements alias searching and ID generation (`[TYPE:3]-[ID:5]-[SLUG:6]`). |
| [src/normalizer/rate_limiter.py](src/normalizer/rate_limiter.py) | No | Token-Bucket rate limiter enforcing Gemini API RPM and TPM quotas. |
| [src/normalizer/gemini_client.py](src/normalizer/gemini_client.py) | No | Gemini SDK client (`google-genai`) handling batched entity normalization, context caching, and 429 adaptive backoff. |
| [src/normalizer/affiliation_normalizer.py](src/normalizer/affiliation_normalizer.py) | No | Normalization coordinator orchestrating local registry lookup and batched Gemini LLM resolution. |
| [src/exporters/matrix_exporter.py](src/exporters/matrix_exporter.py) | No | Generates final matrix CSV files with paper deduplication and institution sorting. |
| `data/canonical_organizations.json` | Persistent Data | Central persistent database of canonical institutional entities, types, and aliases. |
| `data/openalex_sources_cache.json` | Persistent Cache | Resolved mapping of venue acronyms to OpenAlex Source IDs. |

---

## Running Unit Tests

Execute the automated test suite using Python's `unittest` runner:

```bash
python3 -m unittest discover -s tests
```

The test suite covers:
- YAML batch config parsing and exception merging (`test_batch_runner.py`).
- OpenAlex source resolution and filter fallback protection (`test_extractors.py`).
- Registry search, canonical ID generation, and alias matching (`test_registry.py`).
- Token-bucket rate limiting and logger formatting (`test_logger_and_rate_limiter.py`).
- Matrix CSV exporting and paper deduplication (`test_normalizer_and_exporter.py`).
