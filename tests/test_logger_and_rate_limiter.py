import logging
import tempfile
import unittest
from pathlib import Path

from src.logger import setup_logging, LOG_FORMAT
from src.normalizer.rate_limiter import TokenBucketRateLimiter


class TestLoggerAndRateLimiter(unittest.TestCase):

    def test_log_formatting(self):
        # Verify logger format uses square brackets
        self.assertIn("[%(asctime)s]", LOG_FORMAT)
        self.assertIn("[PID:%(process)d]", LOG_FORMAT)
        self.assertIn("[% (name)s]".replace(" ", ""), LOG_FORMAT)
        self.assertIn("[% (levelname)s]".replace(" ", ""), LOG_FORMAT)

    def test_rate_limiter_update(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=30.0)
        self.assertEqual(limiter.capacity, 30.0)

        # Update from initial API query limit
        limiter.update_limit(60.0)
        self.assertEqual(limiter.capacity, 60.0)
        self.assertAlmostEqual(limiter.fill_rate, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()

