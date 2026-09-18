import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.extractors.base import BaseExtractor
from src.extractors.openalex import OpenAlexExtractor, derive_doi_prefix
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
            
            # Pre-populate cache
            cache_path.write_text('{"IROS": "S4363608614"}', encoding="utf-8")
            resolved_id = extractor.resolve_source_id("IROS")
            self.assertEqual(resolved_id, "S4363608614")

    def test_openalex_filter_no_search_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "openalex_sources_cache.json"
            extractor = OpenAlexExtractor(api_key="dummy", mailto="test@example.com", sources_cache_path=cache_path)
            
            # Mock session response to return count=0 for all test queries
            mock_res = MagicMock()
            mock_res.status_code = 200
            mock_res.json.return_value = {"meta": {"count": 0}}
            extractor.session.get = MagicMock(return_value=mock_res)

            params = extractor._determine_filter_params("ICRA", 2024, "S12345", {})
            self.assertNotIn("search", params)
            self.assertIn("filter", params)
            self.assertIn("doi_starts_with:10.1109/icra", params["filter"])

    def test_derive_doi_prefix(self):
        self.assertEqual(derive_doi_prefix("ICRA"), "10.1109/icra")
        self.assertEqual(derive_doi_prefix("ICRA (International Conference...)"), "10.1109/icra")
        self.assertEqual(derive_doi_prefix("T-RO (IEEE Transactions on Robotics)"), "10.1109/tro")
        self.assertEqual(derive_doi_prefix("R-AL (IEEE Robotics...)"), "10.1109/ral")
        self.assertEqual(derive_doi_prefix("R-AL", explicit_prefix="10.1109/lra"), "10.1109/lra")

    def test_sanitize_venue_name(self):
        self.assertEqual(sanitize_venue_name("IROS (IEEE/RSJ International Conference on Intelligent Robots and Systems)"), "IROS")
        self.assertEqual(sanitize_venue_name("ICRA (International Conference on Robotics and Automation)"), "ICRA")
        self.assertEqual(sanitize_venue_name("T-RO (IEEE Transactions on Robotics)"), "T-RO")
        self.assertEqual(sanitize_venue_name("IEEE/RSJ"), "IEEE_RSJ")


if __name__ == "__main__":
    unittest.main()
