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

    def test_object_venue_parsing(self):
        custom_yaml = self.root_path / "object_venues_batch.yaml"
        content = """
venues:
  - search_term: "IEEE/RSJ International Conference on Intelligent Robots and Systems"
    short_name: IROS
  - search_term: "International Conference on Robotics and Automation"
    short_name: ICRA
years: [2024]
source: openalex
"""
        with open(custom_yaml, "w", encoding="utf-8") as f:
            f.write(content)

        config = BatchConfig.from_file(custom_yaml)
        self.assertEqual(config.venues, ["IROS", "ICRA"])
        iros_overrides = config.get_overrides("IROS", 2024)
        self.assertEqual(iros_overrides.get("search_term"), "IEEE/RSJ International Conference on Intelligent Robots and Systems")
        icra_overrides = config.get_overrides("ICRA", 2024)
        self.assertEqual(icra_overrides.get("search_term"), "International Conference on Robotics and Automation")

    def test_multiple_search_terms_parsing(self):
        custom_yaml = self.root_path / "multi_search_batch.yaml"
        content = """
venues:
  - search_term:
      - "European Conference on Computer Vision"
      - "ECCV"
    short_name: ECCV
years: [2024]
source: openalex
        """
        with open(custom_yaml, "w", encoding="utf-8") as f:
            f.write(content)

        config = BatchConfig.from_file(custom_yaml)
        self.assertEqual(config.venues, ["ECCV"])
        eccv_overrides = config.get_overrides("ECCV", 2024)
        self.assertEqual(eccv_overrides.get("search_term"), ["European Conference on Computer Vision", "ECCV"])

    def test_cli_parser_model_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--model", "gemini-3.1-pro", "batch", "-c", "batch.yaml"])
        self.assertEqual(args.model, "gemini-3.1-pro")

        args_sub = parser.parse_args(["batch", "--model", "gemini-3.1-pro"])
        self.assertEqual(args_sub.model, "gemini-3.1-pro")

    def test_bulk_runner_model_propagation(self):
        example_yaml = Path("batch.example.yaml")
        config = BatchConfig.from_file(example_yaml)
        runner = BulkRunner(config=config, model="gemini-3.1-pro")
        self.assertEqual(runner.model, "gemini-3.1-pro")
        self.assertEqual(runner.gemini_client.model, "gemini-3.1-pro")
        self.assertEqual(runner.normalizer.gemini_client.model, "gemini-3.1-pro")


if __name__ == "__main__":
    unittest.main()
