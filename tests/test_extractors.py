import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.extractors.base import BaseExtractor
from src.extractors.openalex import OpenAlexExtractor, normalize_cache_entry
from src.utils import sanitize_venue_name


class DummyExtractor(BaseExtractor):
    def extract(self, venue: str, year: int, force: bool = False):
        return {}


class TestExtractors(unittest.TestCase):

    def test_append_raw_batch_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            extractor = DummyExtractor(output_dir=Path(tmp_dir))
            
            # Test initial batch appending with completed=False
            res1 = extractor.append_raw_batch(
                venue="ICRA",
                year=2024,
                new_papers=[{"paper_id": "p1", "title": "Paper 1", "raw_affiliations": ["MIT"]}],
                next_cursor="cursor_abc",
                completed=False,
            )
            self.assertFalse(res1["completed"])
            self.assertEqual(res1["next_cursor"], "cursor_abc")
            self.assertEqual(res1["total_papers"], 1)

            # Test resuming check
            state = extractor.get_resume_state("ICRA", 2024)
            self.assertFalse(state["completed"])
            self.assertEqual(state["next_cursor"], "cursor_abc")

            # Test marking completed
            res2 = extractor.append_raw_batch(
                venue="ICRA",
                year=2024,
                new_papers=[{"paper_id": "p2", "title": "Paper 2", "raw_affiliations": ["Stanford"]}],
                next_cursor=None,
                completed=True,
            )
            self.assertTrue(res2["completed"])
            self.assertEqual(res2["total_papers"], 2)

    def test_openalex_sources_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "openalex_sources_cache.json"
            extractor = OpenAlexExtractor(api_key="dummy", mailto="test@example.com", sources_cache_path=cache_path)
            
            # Legacy string cache format test
            cache_path.write_text('{"IROS": "S4363608614"}', encoding="utf-8")
            resolved_id = extractor.resolve_source_id("IROS")
            self.assertEqual(resolved_id, "S4363608614")

            # Multi-ID & DOI structured cache format test
            struct_data = {
                "CORL": {
                    "source_ids": ["S4306506823", "S4306499611"],
                    "doi_prefixes": ["10.5555/corl"],
                    "frequency": "annual",
                    "years": {
                        "2023": {
                            "source_ids": ["S4306499611"],
                            "doi_prefixes": ["10.5555/corl2023"]
                        }
                    }
                }
            }
            import json
            cache_path.write_text(json.dumps(struct_data), encoding="utf-8")

            ids_gen, prefs_gen, freq_gen = extractor.resolve_source_info("CORL")
            self.assertEqual(ids_gen, ["S4306506823", "S4306499611"])
            self.assertEqual(prefs_gen, ["10.5555/corl"])
            self.assertEqual(freq_gen, "annual")

            ids_2023, prefs_2023, _ = extractor.resolve_source_info("CORL", year=2023)
            self.assertEqual(ids_2023, ["S4306499611"])
            self.assertEqual(prefs_2023, ["10.5555/corl2023"])

    def test_biennial_off_year_skipping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            cache_path = tmp_p / "openalex_sources_cache.json"
            import json
            cache_path.write_text(json.dumps({
                "ECCV": {
                    "source_ids": ["S4306418318"],
                    "doi_prefixes": [],
                    "frequency": "biennial_even"
                }
            }), encoding="utf-8")

            extractor = OpenAlexExtractor(
                api_key="dummy",
                mailto="test@example.com",
                sources_cache_path=cache_path,
                output_dir=tmp_p
            )

            # Test extracting off-year 2017 (odd year for biennial_even)
            res_2017 = extractor.extract("ECCV", 2017, force=True)
            self.assertTrue(res_2017.get("completed"))
            self.assertEqual(res_2017.get("total_papers"), 0)

    def test_openalex_filter_multi_source_and_doi(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "openalex_sources_cache.json"
            extractor = OpenAlexExtractor(api_key="dummy", mailto="test@example.com", sources_cache_path=cache_path)
            
            # Mock session response to return count=0 for all test queries
            mock_res = MagicMock()
            mock_res.status_code = 200
            mock_res.json.return_value = {"meta": {"count": 0}}
            extractor.session.get = MagicMock(return_value=mock_res)

            params = extractor._determine_filter_params(
                "ICRA", 2024, ["S12345", "S67890"], ["10.1109/icra"], {}
            )
            self.assertNotIn("search", params)
            self.assertIn("filter", params)
            self.assertIn("doi_starts_with:10.1109/icra", params["filter"])

    def test_normalize_cache_entry(self):
        # Legacy string
        self.assertEqual(normalize_cache_entry("S123")["source_ids"], ["S123"])
        # Legacy URL
        self.assertEqual(normalize_cache_entry("https://openalex.org/S123")["source_ids"], ["S123"])
        # Multi ID dict
        res = normalize_cache_entry({"source_ids": ["S1", "S2"], "doi_prefixes": ["10.1109/icra"], "frequency": "biennial_even"})
        self.assertEqual(res["source_ids"], ["S1", "S2"])
        self.assertEqual(res["doi_prefixes"], ["10.1109/icra"])
        self.assertEqual(res["frequency"], "biennial_even")

    def test_sanitize_venue_name(self):
        self.assertEqual(sanitize_venue_name("IROS (IEEE/RSJ International Conference on Intelligent Robots and Systems)"), "IROS")
        self.assertEqual(sanitize_venue_name("ICRA (International Conference on Robotics and Automation)"), "ICRA")
        self.assertEqual(sanitize_venue_name("T-RO (IEEE Transactions on Robotics)"), "T-RO")
        self.assertEqual(sanitize_venue_name("IEEE/RSJ"), "IEEE_RSJ")


if __name__ == "__main__":
    unittest.main()
