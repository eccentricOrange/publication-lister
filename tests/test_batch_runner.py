import tempfile
import unittest
from pathlib import Path

from src.runner.batch_config import BatchConfig
from src.runner.bulk_runner import BulkRunner
from main import build_parser


class TestBatchRunner(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_batch_yaml_example(self):
        example_yaml = Path("batch.yaml.example")
        example_yaml = Path("batch.example.yaml")
        self.assertTrue(example_yaml.exists())

        config = BatchConfig.from_file(example_yaml)
        self.assertEqual(config.venues, ["ICRA", "IROS", "CVPR", "NEURIPS"])
        self.assertEqual(config.years, [2020, 2021, 2022, 2023, 2024, 2025])
        self.assertEqual(config.default_source, "openalex")

        # Test overrides
        iros_overrides = config.get_overrides("IROS", 2025)
        self.assertEqual(iros_overrides.get("openalex_source_id"), "S4363608614")

        cvpr_overrides = config.get_overrides("CVPR", 2021)
        self.assertEqual(cvpr_overrides.get("search_term"), "IEEE/CVF Conference on Computer Vision and Pattern Recognition")

    def test_cli_parser_batch_default_config(self):
        parser = build_parser()
        args = parser.parse_args(["batch"])
        self.assertEqual(args.subcommand, "batch")
        self.assertEqual(args.config, "batch.yaml")

    def test_custom_yaml_config_creation_and_parsing(self):
        custom_yaml = self.root_path / "custom_batch.yaml"
        content = """
venues:
  - ICRA
  - IROS
years:
  start: 2023
  end: 2024
source: openalex
exceptions:
  - venue: ICRA
    years: [2024]
    doi_prefix: "10.1109/icra"
"""
        with open(custom_yaml, "w", encoding="utf-8") as f:
            f.write(content)

        config = BatchConfig.from_file(custom_yaml)
        self.assertEqual(config.venues, ["ICRA", "IROS"])
        self.assertEqual(config.years, [2023, 2024])
        
        icra_2024_overrides = config.get_overrides("ICRA", 2024)
        self.assertEqual(icra_2024_overrides.get("doi_prefix"), "10.1109/icra")


if __name__ == "__main__":
    unittest.main()
