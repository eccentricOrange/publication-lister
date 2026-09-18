import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.cleaner.csv_cleaner import CSVCleaner
from src.registry.organization_registry import OrganizationRegistry


class TestCSVCleaner(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        self.out_dir = self.root_path / "data" / "output"
        self.cleaned_dir = self.root_path / "data" / "cleaned_output"
        self.reg_path = self.root_path / "data" / "canonical_organizations.json"

        # Create registry
        self.seed_data = [
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
                "known_aliases": ["MIT", "CSAIL, MIT", "MIT Media Labs"],
            },
        ]
        self.reg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.reg_path, "w", encoding="utf-8") as f:
            json.dump(self.seed_data, f)

        self.registry = OrganizationRegistry(registry_path=self.reg_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_csv_cleaner_pruning_and_combination(self):
        # Create uncleaned CSV matrix in self.out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        input_csv = self.out_dir / "ICRA_affiliations_2023_2024.csv"

        headers = ["canonical_id", "canonical_name", "entity_type", "2023", "2024", "total"]
        rows = [
            {
                "canonical_id": "UNI-00002-MITCAM",
                "canonical_name": "MIT CSAIL",
                "entity_type": "UNI",
                "2023": "5",
                "2024": "10",
                "total": "15",
            },
            {
                "canonical_id": "UNI-00099-MITMED",
                "canonical_name": "MIT Media Labs",
                "entity_type": "UNI",
                "2023": "3",
                "2024": "2",
                "total": "5",
            },
            {
                "canonical_id": "UNI-00001-STANFD",
                "canonical_name": "Stanford University",
                "entity_type": "UNI",
                "2023": "8",
                "2024": "12",
                "total": "20",
            },
            {
                "canonical_id": "UNKNOWN-001",
                "canonical_name": "Department of Electrical Engineering",
                "entity_type": "UNI",
                "2023": "1",
                "2024": "0",
                "total": "1",
            },
            {
                "canonical_id": "UNKNOWN-002",
                "canonical_name": "UNKNOWN",
                "entity_type": "UNI",
                "2023": "4",
                "2024": "4",
                "total": "8",
            },
        ]

        with open(input_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        # Mock Gemini 3-step response
        mock_gemini = MagicMock()
        mock_gemini.api_key = "dummy_key"
        mock_gemini.rate_limiter = MagicMock()
        
        step1_response_json = {
            "prune_ids": ["UNKNOWN-001", "UNKNOWN-002"],
            "requested_parent_names": ["MIT"],
            "merges": [
                {
                    "target_parent_name": "MIT",
                    "source_ids": ["UNI-00002-MITCAM", "UNI-00099-MITMED"]
                }
            ]
        }

        step3_response_json = {
            "cleaned_rows": [
                {
                    "canonical_id": "UNI-00001-STANFD",
                    "canonical_name": "Stanford University",
                    "entity_type": "UNI",
                    "counts": {"2023": 8, "2024": 12},
                },
                {
                    "canonical_id": "UNI-00002-MITCAM",
                    "canonical_name": "Massachusetts Institute of Technology",
                    "entity_type": "UNI",
                    "counts": {"2023": 8, "2024": 12},
                },
            ]
        }

        mock_gemini._extract_response_text.side_effect = [
            json.dumps(step1_response_json),
            json.dumps(step3_response_json),
        ]
        mock_gemini.client.models.generate_content.return_value = MagicMock()


        # Record modification time of registry file before cleaning
        reg_mtime_before = self.reg_path.stat().st_mtime

        cleaner = CSVCleaner(
            registry=self.registry,
            gemini_client=mock_gemini,
            output_dir=self.cleaned_dir,
        )

        cleaned_csv = cleaner.clean_file(input_csv)

        # 1. Verify cleaned file location
        self.assertTrue(cleaned_csv.exists())
        self.assertEqual(cleaned_csv.parent, self.cleaned_dir)

        # 2. Verify contents of cleaned CSV
        with open(cleaned_csv, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 2)  # MIT combined, Stanford kept, department & UNKNOWN pruned

            # Rows sorted by total descending (both total=20, check entries)
            names = {r["canonical_name"] for r in reader}
            self.assertIn("Stanford University", names)
            self.assertIn("Massachusetts Institute of Technology", names)

            # Verify MIT combined total counts (8+12 = 20)
            mit_row = [r for r in reader if r["canonical_name"] == "Massachusetts Institute of Technology"][0]
            self.assertEqual(mit_row["2023"], "8")
            self.assertEqual(mit_row["2024"], "12")
            self.assertEqual(mit_row["total"], "20")

        # 3. Verify canonical registry file was UNTOUCHED
        reg_mtime_after = self.reg_path.stat().st_mtime
        self.assertEqual(reg_mtime_before, reg_mtime_after)
        with open(self.reg_path, "r", encoding="utf-8") as f:
            reg_content_after = json.load(f)
        self.assertEqual(reg_content_after, self.seed_data)


if __name__ == "__main__":
    unittest.main()

