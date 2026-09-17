import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """
    Thread-safe Rate Limiter with strict inter-request pacing.
    Prevents burst traffic from triggering HTTP 429 rate limit errors.
    """

    def __init__(self, requests_per_minute: float = 6.0, tokens_per_minute: float = 250000.0):
        self.capacity = max(1.0, float(requests_per_minute))
        self.tpm_capacity = max(1000.0, float(tokens_per_minute))
        self.fill_rate = self.capacity / 60.0
        self.min_delay = 60.0 / self.capacity  # 10.0s for 6 RPM
        self.last_request_time = 0.0
        self.lock = threading.Lock()

    def update_limit(self, requests_per_minute: float) -> None:
        """Dynamically updates capacity, fill_rate, and min_delay from API headers."""
        self.update_limits(requests_per_minute=requests_per_minute)

    def update_limits(self, requests_per_minute: Optional[float] = None, tokens_per_minute: Optional[float] = None) -> None:
        """Dynamically updates RPM and TPM capacities from API response headers."""
        with self.lock:
            if requests_per_minute is not None:
                self.capacity = max(1.0, float(requests_per_minute))
                self.fill_rate = self.capacity / 60.0
                self.min_delay = 60.0 / self.capacity
            if tokens_per_minute is not None:
                self.tpm_capacity = max(1000.0, float(tokens_per_minute))
            logger.info(
                f"Updated TokenBucketRateLimiter rate limits: {self.capacity:.1f} RPM "
                f"(min_delay={self.min_delay:.2f}s), {self.tpm_capacity:.0f} TPM"
            )

    def acquire(self) -> None:
        """Blocks until minimum inter-request pacing delay has elapsed."""
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_request_time
            if elapsed < self.min_delay:
                wait_time = self.min_delay - elapsed
                time.sleep(wait_time)
            self.last_request_time = time.monotonic()

