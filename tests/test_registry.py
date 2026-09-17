import json
import re
import tempfile
import unittest
from pathlib import Path

from src.registry.organization_registry import OrganizationRegistry, generate_slug


class TestOrganizationRegistry(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.reg_path = Path(self.temp_dir.name) / "test_canonical_organizations.json"
        
        # Initial seed
        seed_data = [
            {
                "canonical_id": "UNI-00001-STANFD",
                "canonical_name": "Stanford University",
                "entity_type": "UNI",
                "known_aliases": ["Stanford University", "Stanford AI Lab"],
            },
            {
                "canonical_id": "COM-00001-GOOGUS",
                "canonical_name": "Google LLC",
                "entity_type": "COM",
                "known_aliases": ["Google LLC", "Google Brain"],
            },
        ]
        with open(self.reg_path, "w", encoding="utf-8") as f:
            json.dump(seed_data, f)

        self.registry = OrganizationRegistry(registry_path=self.reg_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_canonical_id_format(self):
        id_pattern = re.compile(r"^(UNI|COM|LAB|GOV)-\d{5}-[A-Z0-9]{6}$")
        for entry in self.registry.entries:
            self.assertTrue(
                id_pattern.match(entry["canonical_id"]),
                f"Invalid canonical ID format: {entry['canonical_id']}",
            )

    def test_generate_slug(self):
        slug = generate_slug("University of California, San Diego")
        self.assertEqual(len(slug), 6)
        self.assertEqual(generate_slug("Google LLC"), generate_slug("Google LLC"))

    def test_find_by_string(self):
        match = self.registry.find_by_string("Stanford AI Lab")
        self.assertIsNotNone(match)
        self.assertEqual(match["canonical_id"], "UNI-00001-STANFD")

        match_comp = self.registry.find_by_string("Google Brain")
        self.assertIsNotNone(match_comp)
        self.assertEqual(match_comp["canonical_id"], "COM-00001-GOOGUS")

    def test_register_new_organization(self):
        new_entry = self.registry.register_organization(
            canonical_name="University of California, Los Angeles",
            entity_type="UNI",
            known_aliases=["UCLA"],
        )
        expected_slug = generate_slug("University of California, Los Angeles")
        self.assertEqual(new_entry["canonical_id"], f"UNI-00002-{expected_slug}")
        self.assertEqual(new_entry["entity_type"], "UNI")

        # Verify saved to disk
        reg_reloaded = OrganizationRegistry(registry_path=self.reg_path)
        self.assertIsNotNone(reg_reloaded.find_by_string("UCLA"))


if __name__ == "__main__":
    unittest.main()
