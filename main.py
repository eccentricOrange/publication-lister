import argparse
import logging
import sys
from pathlib import Path

from src.config import DEFAULT_GEMINI_MODEL
from src.extractors.conference_schedule import ConferenceScheduleExtractor
from src.extractors.ieee_xplore import IEEEExtractor
from src.extractors.openalex import OpenAlexExtractor
from src.extractors.scopus import ScopusExtractor
from src.logger import setup_logging
from src.normalizer.affiliation_normalizer import AffiliationNormalizer
from src.registry.organization_registry import OrganizationRegistry
from src.runner.batch_config import BatchConfig
from src.runner.bulk_runner import BulkRunner
from src.cleaner.csv_cleaner import CSVCleaner
from src.exporters.matrix_exporter import MatrixExporter

logger = logging.getLogger("main")


def run_extract(args: argparse.Namespace) -> None:
    venue = args.venue.upper()
    start_year = args.year_start
    end_year = args.year_end
    source = args.source.lower()
    force = args.force
    model = getattr(args, "model", DEFAULT_GEMINI_MODEL)

    logger.info(f"Executing EXTRACT subcommand for venue={venue}, years={start_year}..{end_year}, source={source}, model={model}")

    for year in range(start_year, end_year + 1):
        if source == "openalex":
            extractor = OpenAlexExtractor(model=model)
            extractor.extract(venue, year, force=force)
        elif source == "ieee":
            extractor = IEEEExtractor()
            extractor.extract(venue, year, force=force)
        elif source == "scopus":
            extractor = ScopusExtractor()
            extractor.extract(venue, year, force=force)
        elif source == "schedule":
            extractor = ConferenceScheduleExtractor(
                schedule_url=args.schedule_url,
                schedule_file=Path(args.schedule_file) if args.schedule_file else None,
            )
            extractor.extract(venue, year, force=force)
        elif source == "all":
            extracted = False
            # 1. Try OpenAlex first (free, open, high rate limit)
            try:
                extractor = OpenAlexExtractor(model=model)
                extractor.extract(venue, year, force=force)
                extracted = True
            except Exception as e:
                logger.warning(f"OpenAlex extraction failed for {venue} {year}: {e}. Trying IEEE Xplore...")

            # 2. Try IEEE Xplore
            if not extracted:
                try:
                    extractor = IEEEExtractor()
                    extractor.extract(venue, year, force=force)
                    extracted = True
                except Exception as e:
                    logger.warning(f"IEEE extraction failed for {venue} {year}: {e}. Trying Scopus...")

            # 3. Try Scopus
            if not extracted:
                try:
                    extractor = ScopusExtractor()
                    extractor.extract(venue, year, force=force)
                    extracted = True
                except Exception as e:
                    logger.warning(f"Scopus extraction failed for {venue} {year}: {e}. Trying schedule parser...")

            # 4. Try Schedule Parser
            if not extracted and (args.schedule_file or args.schedule_url):
                extractor = ConferenceScheduleExtractor(
                    schedule_url=args.schedule_url,
                    schedule_file=Path(args.schedule_file) if args.schedule_file else None,
                )
                extractor.extract(venue, year, force=force)
                extracted = True

            if not extracted:
                raise RuntimeError(f"All extraction sources failed for {venue} {year}")


def run_normalize(args: argparse.Namespace) -> None:
    venue = args.venue.upper()
    start_year = args.year_start
    end_year = args.year_end
    force = args.force
    model = getattr(args, "model", DEFAULT_GEMINI_MODEL)

    logger.info(f"Executing NORMALIZE subcommand for venue={venue}, years={start_year}..{end_year}, model={model}")

    registry = OrganizationRegistry()
    normalizer = AffiliationNormalizer(registry=registry, model=model)

    for year in range(start_year, end_year + 1):
        normalizer.normalize_venue_year(venue, year, force=force)


def run_export(args: argparse.Namespace) -> None:
    venue = args.venue.upper()
    start_year = args.year_start
    end_year = args.year_end
    output_val = getattr(args, "output", None)
    output_path = Path(output_val) if output_val else None

    logger.info(f"Executing EXPORT subcommand for venue={venue}, years={start_year}..{end_year}")

    registry = OrganizationRegistry()
    exporter = MatrixExporter(registry=registry)
    csv_file = exporter.export_matrix(venue, start_year, end_year, output_path=output_path)
    logger.info(f"Export complete: {csv_file}")


def run_clean(args: argparse.Namespace) -> None:
    input_val = getattr(args, "input", None)
    output_val = getattr(args, "output", None)
    clean_all_flag = getattr(args, "all", False)
    venue = getattr(args, "venue", None)
    start_year = getattr(args, "year_start", None)
    end_year = getattr(args, "year_end", None)
    force = getattr(args, "force", False)
    config_val = getattr(args, "config", None)
    config_path = Path(config_val) if config_val else None
    model = getattr(args, "model", DEFAULT_GEMINI_MODEL)

    logger.info(f"Executing CLEAN subcommand with model={model}")
    cleaner = CSVCleaner(model=model)

    if input_val:
        in_p = Path(input_val)
        out_p = Path(output_val) if output_val else None
        if in_p.is_dir():
            cleaner.clean_all(input_dir=in_p, output_dir=out_p, force=force, config_path=config_path)
        else:
            cleaner.clean_file(in_p, output_csv_path=out_p, force=force)
    elif clean_all_flag or (not venue and not start_year):
        cleaner.clean_all(force=force, config_path=config_path)
    elif venue and start_year and end_year:
        from src.utils import sanitize_venue_name
        clean_venue = sanitize_venue_name(venue)
        from src.config import OUTPUT_DATA_DIR
        target_input = OUTPUT_DATA_DIR / f"{clean_venue}_affiliations_{start_year}_{end_year}.csv"
        target_output = Path(output_val) if output_val else None
        cleaner.clean_file(target_input, output_csv_path=target_output, force=force)
    else:
        logger.error("Must specify --input, --all, or venue with --year-start and --year-end for clean subcommand.")
        sys.exit(1)


def run_build_visualisation(args: argparse.Namespace) -> None:
    from scripts.build_site_data import build_site_data
    logger.info("Executing BUILD-VISUALISATION subcommand...")
    config_val = getattr(args, "config", None)
    config_path = Path(config_val) if config_val else None
    build_site_data(config_path=config_path)
    logger.info("Visualisation dataset manifest built successfully.")


def run_pipeline(args: argparse.Namespace) -> None:
    logger.info("Executing FULL PIPELINE flow...")
    run_extract(args)
    run_normalize(args)
    run_export(args)
    run_clean(args)
    if getattr(args, "build_visualisation", False):
        run_build_visualisation(args)
    logger.info("Full pipeline completed successfully.")


def run_batch(args: argparse.Namespace) -> None:
    config_path = Path(args.config) if args.config else Path("batch.yaml")
    force = getattr(args, "force", False)
    model = getattr(args, "model", DEFAULT_GEMINI_MODEL)

    logger.info(f"Executing BATCH subcommand with YAML config: {config_path.resolve()}, model={model}")
    config = BatchConfig.from_file(config_path)
    runner = BulkRunner(config=config, model=model)
    runner.run_all(force=force)
    if getattr(args, "build_visualisation", False):
        run_build_visualisation(args)
    logger.info("Batch pipeline execution completed successfully.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="affiliation-tracker",
        description="Academic Conference & Journal Affiliation Tracker",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_GEMINI_MODEL, help=f"Gemini LLM model name (defaults to '{DEFAULT_GEMINI_MODEL}')")

    subparsers = parser.add_subparsers(dest="subcommand", required=True, help="Subcommands")

    # Common arguments helper
    def add_common_args(subparser):
        subparser.add_argument("--venue", "-v_name", type=str, required=True, help="Venue name (e.g., ICRA, IROS, CVPR)")
        subparser.add_argument("--year-start", type=int, required=True, help="Start publication year (e.g., 2017)")
        subparser.add_argument("--year-end", type=int, required=True, help="End publication year (e.g., 2026)")
        subparser.add_argument("--model", "-m", type=str, default=argparse.SUPPRESS, help=f"Gemini LLM model name (defaults to '{DEFAULT_GEMINI_MODEL}')")
        subparser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging")

    # 1. Extract subcommand
    p_extract = subparsers.add_parser("extract", help="Extract raw paper & affiliation data")
    add_common_args(p_extract)
    p_extract.add_argument("--source", type=str, choices=["ieee", "openalex", "scopus", "schedule", "all"], default="ieee", help="Data extractor source")
    p_extract.add_argument("--schedule-file", type=str, default=None, help="Path to local HTML schedule file")
    p_extract.add_argument("--schedule-url", type=str, default=None, help="URL to online conference schedule")
    p_extract.add_argument("--force", action="store_true", help="Force re-extraction ignoring cache")

    # 2. Normalize subcommand
    p_norm = subparsers.add_parser("normalize", help="Normalize raw affiliations with Gemini LLM")
    add_common_args(p_norm)
    p_norm.add_argument("--force", action="store_true", help="Force re-normalization ignoring cached normalized data")

    # 3. Export subcommand
    p_export = subparsers.add_parser("export", help="Aggregate normalized data into CSV matrix")
    add_common_args(p_export)
    p_export.add_argument("--output", "-o", type=str, default=None, help="Output CSV path")

    # 4. Clean subcommand
    p_clean = subparsers.add_parser("clean", help="Clean matrix CSVs with Gemini LLM (prune useless/department entries, merge sub-entities)")
    p_clean.add_argument("--input", "-i", type=str, default=None, help="Input CSV file or directory path")
    p_clean.add_argument("--output", "-o", type=str, default=None, help="Output cleaned CSV file or directory path")
    p_clean.add_argument("--venue", "-v_name", type=str, default=None, help="Venue name (optional)")
    p_clean.add_argument("--year-start", type=int, default=None, help="Start publication year (optional)")
    p_clean.add_argument("--year-end", type=int, default=None, help="End publication year (optional)")
    p_clean.add_argument("--all", action="store_true", help="Clean all CSV matrix files in output directory")
    p_clean.add_argument("--force", action="store_true", help="Force re-cleaning ignoring existing cleaned files and checkpoints")
    p_clean.add_argument("--config", "-c", type=str, default=None, help="Path to batch.yaml configuration file (defaults to batch.yaml if present)")
    p_clean.add_argument("--model", "-m", type=str, default=argparse.SUPPRESS, help=f"Gemini LLM model name (defaults to '{DEFAULT_GEMINI_MODEL}')")
    p_clean.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging")

    # 5. Pipeline subcommand
    p_pipe = subparsers.add_parser("pipeline", help="Run full extraction, normalization, export, and clean pipeline")
    add_common_args(p_pipe)
    p_pipe.add_argument("--source", type=str, choices=["ieee", "openalex", "scopus", "schedule", "all"], default="ieee", help="Data extractor source")
    p_pipe.add_argument("--schedule-file", type=str, default=None, help="Path to local HTML schedule file")
    p_pipe.add_argument("--schedule-url", type=str, default=None, help="URL to online conference schedule")
    p_pipe.add_argument("--output", "-o", type=str, default=None, help="Output CSV path")
    p_pipe.add_argument("--force", action="store_true", help="Force re-run of all pipeline steps")
    p_pipe.add_argument("--build-visualisation", "--build-vis", action="store_true", help="Automatically generate visualization JSON manifest after pipeline completion")

    # 6. Batch subcommand
    p_batch = subparsers.add_parser("batch", help="Run multi-venue bulk pipeline using YAML configuration")
    p_batch.add_argument("--config", "-c", type=str, default="batch.yaml", help="Path to batch.yaml configuration file (defaults to batch.yaml in root)")
    p_batch.add_argument("--force", action="store_true", help="Force re-run of all bulk extraction and normalization steps")
    p_batch.add_argument("--model", "-m", type=str, default=argparse.SUPPRESS, help=f"Gemini LLM model name (defaults to '{DEFAULT_GEMINI_MODEL}')")
    p_batch.add_argument("--build-visualisation", "--build-vis", action="store_true", help="Automatically generate visualization JSON manifest after batch completion")
    p_batch.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging")

    # 7. Build Visualisation subcommand
    p_vis = subparsers.add_parser("build-visualisation", aliases=["visualize", "build-vis"], help="Build web visualization dataset manifest in docs/data/")
    p_vis.add_argument("--config", "-c", type=str, default=None, help="Path to batch.yaml configuration file (defaults to batch.yaml if present)")
    p_vis.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging")

    return parser


def cli() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Configure centralized logging at entrypoint
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    setup_logging(level=log_level)

    try:
        if args.subcommand == "extract":
            run_extract(args)
        elif args.subcommand == "normalize":
            run_normalize(args)
        elif args.subcommand == "export":
            run_export(args)
        elif args.subcommand == "clean":
            run_clean(args)
        elif args.subcommand == "pipeline":
            run_pipeline(args)
        elif args.subcommand == "batch":
            run_batch(args)
        elif args.subcommand in ("build-visualisation", "visualize", "build-vis"):
            run_build_visualisation(args)
    except Exception as e:
        logger.error(f"Execution failed on subcommand '{args.subcommand}'", exc_info=True)
        sys.exit(1)



if __name__ == "__main__":
    cli()
