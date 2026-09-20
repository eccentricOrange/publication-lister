import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.extractors.openalex import OpenAlexExtractor
from src.extractors.ieee_xplore import IEEEExtractor
from src.extractors.scopus import ScopusExtractor
from src.normalizer.affiliation_normalizer import AffiliationNormalizer
from src.exporters.matrix_exporter import MatrixExporter
from src.registry.organization_registry import OrganizationRegistry


class TestNormalizerAndExporter(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        
        self.raw_dir = self.root_path / "data" / "raw"
        self.norm_dir = self.root_path / "data" / "normalized"
        self.out_dir = self.root_path / "data" / "output"
        self.reg_path = self.root_path / "data" / "canonical_organizations.json"

        # Create registry
        seed_data = [
            {
                "canonical_id": "UNI-00001-STANFD",
                "canonical_name": "Stanford University",
                "entity_type": "UNI",
                "known_aliases": ["Stanford University", "Stanford CS"],
            },
            {
                "canonical_id": "UNI-00002-MITCAM",
                "canonical_name": "Massachusetts Institute of Technology",
                "entity_type": "UNI",
                "known_aliases": ["MIT", "CSAIL, MIT"],
            },
        ]
        self.reg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.reg_path, "w", encoding="utf-8") as f:
            json.dump(seed_data, f)

        self.registry = OrganizationRegistry(registry_path=self.reg_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_strict_exception_on_missing_api_keys(self):
        # Verify OpenAlexExtractor raises ValueError when api_key and mailto are empty
        openalex_ext = OpenAlexExtractor(api_key="", mailto="")
        with self.assertRaises(ValueError):
            openalex_ext.extract("ICRA", 2024)

        # Verify IEEEExtractor raises ValueError when key is empty
        extractor = IEEEExtractor(api_key="")
        with self.assertRaises(ValueError):
            extractor.extract("ICRA", 2024)

        # Verify ScopusExtractor raises ValueError when key is empty
        scopus_ext = ScopusExtractor(api_key="")
        with self.assertRaises(ValueError):
            scopus_ext.extract("ICRA", 2024)

    def test_exporter_within_paper_deduplication_and_matrix(self):
        # Create normalized dataset for 2023 and 2024
        self.norm_dir.mkdir(parents=True, exist_ok=True)

        norm_2023 = {
            "venue": "ICRA",
            "year": 2023,
            "total_papers": 2,
            "papers": [
                {
                    "paper_id": "P1",
                    "title": "Paper 1",
                    # 2 authors from Stanford (deduplicated to 1 count for P1) and 1 from MIT
                    "canonical_ids": ["UNI-00001-STANFD", "UNI-00002-MITCAM"],
                },
                {
                    "paper_id": "P2",
                    "title": "Paper 2",
                    "canonical_ids": ["UNI-00001-STANFD"],
                },
            ],
        }

        norm_2024 = {
            "venue": "ICRA",
            "year": 2024,
            "total_papers": 1,
            "papers": [
                {
                    "paper_id": "P3",
                    "title": "Paper 3",
                    "canonical_ids": ["UNI-00001-STANFD"],
                },
            ],
        }

        with open(self.norm_dir / "ICRA_2023_normalized.json", "w", encoding="utf-8") as f:
            json.dump(norm_2023, f)

        with open(self.norm_dir / "ICRA_2024_normalized.json", "w", encoding="utf-8") as f:
            json.dump(norm_2024, f)

        exporter = MatrixExporter(
            registry=self.registry,
            normalized_dir=self.norm_dir,
            output_dir=self.out_dir,
        )

        csv_file = exporter.export_matrix("ICRA", 2023, 2024)
        self.assertTrue(csv_file.exists())

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 2)
            
            # Stanford should be row 1 with 2 in 2023, 1 in 2024, total = 3
            row_stanford = reader[0]
            self.assertEqual(row_stanford["canonical_id"], "UNI-00001-STANFD")
            self.assertEqual(row_stanford["2023"], "2")
            self.assertEqual(row_stanford["2024"], "1")
            self.assertEqual(row_stanford["total"], "3")

            # MIT should be row 2 with 1 in 2023, 0 in 2024, total = 1
            row_mit = reader[1]
            self.assertEqual(row_mit["canonical_id"], "UNI-00002-MITCAM")
            self.assertEqual(row_mit["2023"], "1")
            self.assertEqual(row_mit["2024"], "0")
            self.assertEqual(row_mit["total"], "1")

    def test_pause_and_resume_state(self):
        extractor = IEEEExtractor(api_key="TEST_KEY", output_dir=self.raw_dir)
        
        # Save partial state (paused after page 1, next_page=201, completed=False)
        batch_1 = [{"paper_id": f"P_{i}", "title": f"Paper {i}", "raw_affiliations": ["Stanford"]} for i in range(1, 201)]
        extractor.append_raw_batch("ICRA", 2022, batch_1, completed=False, next_page=201)
        
        self.assertFalse(extractor.is_cached("ICRA", 2022))
        
        # Verify file on disk contains pause state
        raw_file = extractor.get_raw_file_path("ICRA", 2022)
        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["total_papers"], 200)
            self.assertEqual(data["next_page"], 201)
            self.assertFalse(data["completed"])

        # Mark completed
        extractor.append_raw_batch("ICRA", 2022, [], completed=True)
        self.assertTrue(extractor.is_cached("ICRA", 2022))

    def test_gemini_client_retry_configuration(self):
        from src.normalizer.gemini_client import GeminiClient
        client = GeminiClient(api_key="dummy_key_for_testing")
        genai_client = client.client
        self.assertIsNotNone(genai_client._api_client._http_options)
        retry_opts = genai_client._api_client._http_options.retry_options
        self.assertIsNotNone(retry_opts)
        self.assertEqual(retry_opts.attempts, 2)
        self.assertIn(503, retry_opts.http_status_codes)
        self.assertNotIn(504, retry_opts.http_status_codes)

    def test_gemini_client_normalize_batch_timeout_split(self):
        from unittest.mock import MagicMock
        from src.normalizer.gemini_client import GeminiClient

        client = GeminiClient(api_key="dummy_key_for_testing")
        client._client = MagicMock()

        # Simulate 504 Timeout on first call with 20 strings, then success on sub-batches of 10
        raw_strings = [f"Affiliation_{i}" for i in range(20)]

        def side_effect(model, contents, config):
            payload_str = contents[1]
            if "Affiliation_0" in payload_str and "Affiliation_19" in payload_str:
                raise Exception("504 DEADLINE_EXCEEDED: Deadline expired")
            # Sub-batch 1
            if "Affiliation_0" in payload_str:
                mock_resp = MagicMock()
                mock_resp.text = json.dumps({"resolutions": {f"Affiliation_{i}": f"UNI-0000{i}" for i in range(10)}})
                return mock_resp
            # Sub-batch 2
            mock_resp = MagicMock()
            mock_resp.text = json.dumps({"resolutions": {f"Affiliation_{i}": f"UNI-0000{i}" for i in range(10, 20)}})
            return mock_resp

        client.client.models.generate_content.side_effect = side_effect
        resolutions = client.normalize_batch(raw_strings)

        self.assertEqual(len(resolutions), 20)
        self.assertEqual(resolutions["Affiliation_0"]["canonical_id"], "UNI-00000")
        self.assertEqual(resolutions["Affiliation_19"]["canonical_id"], "UNI-000019")



    def test_parse_gemini_json(self):
        from src.normalizer.gemini_client import parse_gemini_json

        # Test 1: Clean JSON
        res1 = parse_gemini_json('{"resolutions": {"Stanford": "UNI-00001"}}')
        self.assertEqual(res1["resolutions"]["Stanford"], "UNI-00001")

        # Test 2: Markdown fence wrapped
        res2 = parse_gemini_json('```json\n{"resolutions": {"MIT": "UNI-00002"}}\n```')
        self.assertEqual(res2["resolutions"]["MIT"], "UNI-00002")

        # Test 3: Trailing comma before closing brace
        res3 = parse_gemini_json('{"resolutions": {"Google": "COM-00001",}}')
        self.assertEqual(res3["resolutions"]["Google"], "COM-00001")

        # Test 4: Extra prose surrounding JSON block
        res4 = parse_gemini_json('Here is the JSON response:\n{"resolutions": {"Meta": "COM-00002"}}\nHope this helps!')
        self.assertEqual(res4["resolutions"]["Meta"], "COM-00002")

    def test_gemini_client_normalize_batch_json_error_split(self):
        from unittest.mock import MagicMock
        from src.normalizer.gemini_client import GeminiClient

        client = GeminiClient(api_key="dummy_key_for_testing")
        client._client = MagicMock()

        raw_strings = [f"Affiliation_{i}" for i in range(10)]

        def side_effect(model, contents, config):
            payload_str = contents[1]
            if "Affiliation_0" in payload_str and "Affiliation_9" in payload_str:
                # Return unparseable malformed JSON for full 10-string batch
                mock_resp = MagicMock()
                mock_resp.text = '{"resolutions": {"Affiliation_0": "UNI-0001", invalid_json_here}}'
                return mock_resp
            # Sub-batch 1
            if "Affiliation_0" in payload_str:
                mock_resp = MagicMock()
                mock_resp.text = json.dumps({"resolutions": {f"Affiliation_{i}": f"UNI-0000{i}" for i in range(5)}})
                return mock_resp
            # Sub-batch 2
            mock_resp = MagicMock()
            mock_resp.text = json.dumps({"resolutions": {f"Affiliation_{i}": f"UNI-0000{i}" for i in range(5, 10)}})
            return mock_resp

        client.client.models.generate_content.side_effect = side_effect
        resolutions = client.normalize_batch(raw_strings)

        self.assertEqual(len(resolutions), 10)
        self.assertEqual(resolutions["Affiliation_0"]["canonical_id"], "UNI-00000")
        self.assertEqual(resolutions["Affiliation_9"]["canonical_id"], "UNI-00009")


if __name__ == "__main__":
    unittest.main()



