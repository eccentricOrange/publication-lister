import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """
    Thread-safe Token Bucket Rate Limiter.
    Dynamically initialized from API rate limit query headers on startup.
    """

    def __init__(self, requests_per_minute: float = 15.0):
        self.capacity = max(1.0, float(requests_per_minute))
        self.fill_rate = self.capacity / 60.0  # tokens per second
        self.tokens = self.capacity
        self.last_update = time.monotonic()
        self.lock = threading.Lock()

    def update_limit(self, requests_per_minute: float) -> None:
        """Dynamically updates capacity and fill rate from startup API headers."""
        with self.lock:
            self.capacity = max(1.0, float(requests_per_minute))
            self.fill_rate = self.capacity / 60.0
            self.tokens = min(self.tokens, self.capacity)
            logger.info(f"Updated TokenBucketRateLimiter rate limit to {self.capacity:.1f} RPM")

    def acquire(self) -> None:
        """Blocks until a token is available for consumption."""
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now

                self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                else:
                    needed = 1.0 - self.tokens
                    wait_time = needed / self.fill_rate

            time.sleep(max(0.05, wait_time))

