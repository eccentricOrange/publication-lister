import tempfile
import unittest
from pathlib import Path
from src.extractors.base import BaseExtractor


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
        from src.extractors.openalex import OpenAlexExtractor
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "openalex_sources_cache.json"
            extractor = OpenAlexExtractor(api_key="dummy", mailto="test@example.com", sources_cache_path=cache_path)
            
            # Pre-populate cache
            cache_path.write_text('{"IROS": "S4363608614"}', encoding="utf-8")
            resolved_id = extractor.resolve_source_id("IROS")
            self.assertEqual(resolved_id, "S4363608614")


if __name__ == "__main__":
    unittest.main()

