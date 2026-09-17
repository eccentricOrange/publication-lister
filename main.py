import argparse
import logging
import sys
from pathlib import Path

from src.logger import setup_logging
from src.extractors.ieee_xplore import IEEEExtractor
from src.extractors.scopus import ScopusExtractor
from src.extractors.conference_schedule import ConferenceScheduleExtractor
from src.registry.organization_registry import OrganizationRegistry
from src.normalizer.affiliation_normalizer import AffiliationNormalizer
from src.exporters.matrix_exporter import MatrixExporter

logger = logging.getLogger("main")


def run_extract(args: argparse.Namespace) -> None:
    venue = args.venue.upper()
    start_year = args.year_start
    end_year = args.year_end
    source = args.source.lower()
    force = args.force

    logger.info(f"Executing EXTRACT subcommand for venue={venue}, years={start_year}..{end_year}, source={source}")

    for year in range(start_year, end_year + 1):
        if source == "ieee":
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
            # Attempt extractors in order
            extracted = False
            try:
                extractor = IEEEExtractor()
                extractor.extract(venue, year, force=force)
                extracted = True
            except Exception as e:
                logger.warning(f"IEEE extraction failed for {venue} {year}: {e}. Trying Scopus...")

            if not extracted:
                try:
                    extractor = ScopusExtractor()
                    extractor.extract(venue, year, force=force)
                    extracted = True
                except Exception as e:
                    logger.warning(f"Scopus extraction failed for {venue} {year}: {e}. Trying schedule parser...")

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

    logger.info(f"Executing NORMALIZE subcommand for venue={venue}, years={start_year}..{end_year}")

    registry = OrganizationRegistry()
    normalizer = AffiliationNormalizer(registry=registry)

    for year in range(start_year, end_year + 1):
        normalizer.normalize_venue_year(venue, year, force=force)


def run_export(args: argparse.Namespace) -> None:
    venue = args.venue.upper()
    start_year = args.year_start
    end_year = args.year_end
    output_path = Path(args.output) if args.output else None

    logger.info(f"Executing EXPORT subcommand for venue={venue}, years={start_year}..{end_year}")

    registry = OrganizationRegistry()
    exporter = MatrixExporter(registry=registry)
    csv_file = exporter.export_matrix(venue, start_year, end_year, output_path=output_path)
    logger.info(f"Export complete: {csv_file}")


def run_pipeline(args: argparse.Namespace) -> None:
    logger.info("Executing FULL PIPELINE flow...")
    run_extract(args)
    run_normalize(args)
    run_export(args)
    logger.info("Full pipeline completed successfully.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="affiliation-tracker",
        description="Academic Conference & Journal Affiliation Tracker",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging")

    subparsers = parser.add_subparsers(dest="subcommand", required=True, help="Subcommands")

    # Common arguments helper
    def add_common_args(subparser):
        subparser.add_argument("--venue", "-v_name", type=str, required=True, help="Venue name (e.g., ICRA, IROS, CVPR)")
        subparser.add_argument("--year-start", type=int, required=True, help="Start publication year (e.g., 2017)")
        subparser.add_argument("--year-end", type=int, required=True, help="End publication year (e.g., 2026)")

    # 1. Extract subcommand
    p_extract = subparsers.add_parser("extract", help="Extract raw paper & affiliation data")
    add_common_args(p_extract)
    p_extract.add_argument("--source", type=str, choices=["ieee", "scopus", "schedule", "all"], default="ieee", help="Data extractor source")
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

    # 4. Pipeline subcommand
    p_pipe = subparsers.add_parser("pipeline", help="Run full extraction, normalization, and export pipeline")
    add_common_args(p_pipe)
    p_pipe.add_argument("--source", type=str, choices=["ieee", "scopus", "schedule", "all"], default="ieee", help="Data extractor source")
    p_pipe.add_argument("--schedule-file", type=str, default=None, help="Path to local HTML schedule file")
    p_pipe.add_argument("--schedule-url", type=str, default=None, help="URL to online conference schedule")
    p_pipe.add_argument("--force", action="store_true", help="Force re-run of all pipeline steps")

    return parser


def cli() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Configure centralized logging at entrypoint
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    try:
        if args.subcommand == "extract":
            run_extract(args)
        elif args.subcommand == "normalize":
            run_normalize(args)
        elif args.subcommand == "export":
            run_export(args)
        elif args.subcommand == "pipeline":
            run_pipeline(args)
    except Exception as e:
        logger.error(f"Execution failed on subcommand '{args.subcommand}'", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()

