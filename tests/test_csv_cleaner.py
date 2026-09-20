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

        # Mock Gemini 2-Pass responses
        mock_gemini = MagicMock()
        mock_gemini.api_key = "dummy_key"
        mock_gemini.rate_limiter = MagicMock()

        pass1_json = {
            "problematic_names": [
                "Department of Electrical Engineering",
                "UNKNOWN",
                "MIT CSAIL",
                "MIT Media Labs"
            ]
        }

        pass2_json = {
            "prune_ids": ["UNKNOWN-001", "UNKNOWN-002"],
            "merges": [
                {
                    "target_parent_name": "MIT",
                    "source_ids": ["UNI-00002-MITCAM", "UNI-00099-MITMED"]
                }
            ],
            "type_fixes": {}
        }

        def side_effect(model, contents, config):
            prompt_text = contents[0] if isinstance(contents, list) else ""
            mock_resp = MagicMock()
            if "triage assistant" in prompt_text:
                mock_resp.text = json.dumps(pass1_json)
            else:
                mock_resp.text = json.dumps(pass2_json)
            return mock_resp

        mock_gemini._extract_response_text.side_effect = lambda resp: resp.text
        mock_gemini.client.models.generate_content.side_effect = side_effect

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

    def test_csv_cleaner_checkpoint_resume_and_skip(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        input_csv = self.out_dir / "TEST_affiliations_2023_2024.csv"
        headers = ["canonical_id", "canonical_name", "entity_type", "2023", "total"]
        rows = [
            {"canonical_id": "UNI-00001-STANFD", "canonical_name": "Stanford University", "entity_type": "UNI", "2023": "8", "total": "8"},
        ]
        with open(input_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        output_csv = self.cleaned_dir / "TEST_affiliations_2023_2024.csv"
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        output_csv.write_text("dummy output")

        mock_gemini = MagicMock()
        cleaner = CSVCleaner(registry=self.registry, gemini_client=mock_gemini, output_dir=self.cleaned_dir)

        # 1. Without force, it should skip when target output already exists
        res = cleaner.clean_file(input_csv, force=False)
        self.assertEqual(res, output_csv)
        self.assertEqual(output_csv.read_text(), "dummy output")
        mock_gemini.client.models.generate_content.assert_not_called()

        # 2. With force=True, it re-runs and cleans checkpoint file upon completion
        ckpt_file = self.cleaned_dir / f".checkpoint_{input_csv.name}.json"
        ckpt_data = {
            "pass1_problematic_names": [],
            "pass2_prune_ids": [],
            "pass2_merges": [],
            "pass2_type_fixes": {},
            "pass1_completed_chunks": [0],
            "pass2_completed_chunks": [0],
        }
        ckpt_file.write_text(json.dumps(ckpt_data))

        res_forced = cleaner.clean_file(input_csv, force=True)
        self.assertEqual(res_forced, output_csv)
        self.assertFalse(ckpt_file.exists())  # Checkpoint file deleted after successful run

    def test_clean_all_obeys_batch_yaml(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        icra_csv = self.out_dir / "ICRA_affiliations_2023_2024.csv"
        iros_csv = self.out_dir / "IROS_affiliations_2023_2024.csv"

        headers = ["canonical_id", "canonical_name", "entity_type", "2023", "total"]
        row = {"canonical_id": "UNI-00001-STANFD", "canonical_name": "Stanford", "entity_type": "UNI", "2023": "1", "total": "1"}

        for p in [icra_csv, iros_csv]:
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=headers)
                w.writeheader()
                w.writerow(row)

        custom_yaml = self.root_path / "batch.yaml"
        custom_yaml.write_text("venues:\n  - ICRA\nyears: [2023, 2024]\n")

        mock_gemini = MagicMock()
        mock_gemini.api_key = None  # Will copy file without calling Gemini

        cleaner = CSVCleaner(registry=self.registry, gemini_client=mock_gemini, output_dir=self.cleaned_dir)
        cleaned_paths = cleaner.clean_all(input_dir=self.out_dir, config_path=custom_yaml)

        cleaned_names = [p.name for p in cleaned_paths]
        self.assertIn("ICRA_affiliations_2023_2024.csv", cleaned_names)
        self.assertNotIn("IROS_affiliations_2023_2024.csv", cleaned_names)

    def test_csv_cleaner_refine_and_recursive_convergence(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        input_csv = self.out_dir / "REFINE_affiliations_2023_2024.csv"
        cleaned_csv = self.cleaned_dir / "REFINE_affiliations_2023_2024.csv"

        headers = ["canonical_id", "canonical_name", "entity_type", "2023", "total"]
        raw_rows = [
            {"canonical_id": "UNI-00001-STANFD", "canonical_name": "Stanford University", "entity_type": "UNI", "2023": "5", "total": "5"},
        ]
        with open(input_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(raw_rows)

        # Existing cleaned CSV file
        cleaned_rows = [
            {"canonical_id": "UNI-00001-STANFD", "canonical_name": "Stanford University", "entity_type": "UNI", "2023": "5", "total": "5"},
            {"canonical_id": "UNKNOWN-001", "canonical_name": "Department of Physics", "entity_type": "UNI", "2023": "1", "total": "1"},
        ]
        with open(cleaned_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(cleaned_rows)

        mock_gemini = MagicMock()
        mock_gemini.api_key = "dummy_key"
        mock_gemini.rate_limiter = MagicMock()

        # Pass 1: triage department on pass 1, 0 problematic on pass 2
        pass1_json_1 = {"problematic_names": ["Department of Physics"]}
        pass2_json_1 = {"prune_ids": ["UNKNOWN-001"], "merges": [], "type_fixes": {}}
        pass1_json_2 = {"problematic_names": []}  # pass 2 of recursion -> 0 problematic names -> converge

        call_count = {"p1": 0}
        def side_effect(model, contents, config):
            prompt_text = contents[0] if isinstance(contents, list) else ""
            mock_resp = MagicMock()
            if "triage assistant" in prompt_text:
                call_count["p1"] += 1
                if call_count["p1"] == 1:
                    mock_resp.text = json.dumps(pass1_json_1)
                else:
                    mock_resp.text = json.dumps(pass1_json_2)
            else:
                mock_resp.text = json.dumps(pass2_json_1)
            return mock_resp

        mock_gemini._extract_response_text.side_effect = lambda resp: resp.text
        mock_gemini.client.models.generate_content.side_effect = side_effect

        cleaner = CSVCleaner(registry=self.registry, gemini_client=mock_gemini, output_dir=self.cleaned_dir)

        # Calling clean_file with refine=True should use cleaned_csv as starting point and prune UNKNOWN-001
        res = cleaner.clean_file(input_csv, force=False, refine=True)
        self.assertEqual(res, cleaned_csv)

        with open(cleaned_csv, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 1)
            self.assertEqual(reader[0]["canonical_name"], "Stanford University")


if __name__ == "__main__":
    unittest.main()
