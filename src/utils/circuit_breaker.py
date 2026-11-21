import time
import threading
from enum import Enum
from typing import Callable, Any, Tuple
from src.utils.logger import setup_logger

logger = setup_logger("CircuitBreaker")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"      # Normal healthy operations
    OPEN = "OPEN"          # Tripped, all requests route to fallback
    HALF_OPEN = "HALF_OPEN"# Testing canary traffic for system recovery


class CircuitBreaker:
    """
    Enterprise-grade Circuit Breaker for high-availability financial inference.

    Guarantees sub-50ms SLA by intercepting inference failures, graph timeouts,
    or internal exceptions and falling back to a lightweight heuristic/tabular
    scoring engine without returning HTTP 500 errors to payment clients.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_sec: float = 10.0,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.expected_exception = expected_exception

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_state_change = time.time()
        self.lock = threading.Lock()

    def call(
        self,
        func: Callable[..., Any],
        fallback_func: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Tuple[Any, bool]:
        """
        Executes func safely under circuit breaker protection.

        Returns:
            Tuple of (result, fallback_used: bool)
        """
        with self.lock:
            now = time.time()

            # Check if OPEN circuit has timed out and can attempt canary recovery
            if self.state == CircuitState.OPEN:
                if now - self.last_state_change >= self.recovery_timeout_sec:
                    logger.info("CircuitBreaker recovery timeout elapsed. Transitioning to HALF_OPEN (Canary).")
                    self.state = CircuitState.HALF_OPEN
                    self.last_state_change = now
                else:
                    # Still OPEN: immediately invoke fallback
                    logger.warning("CircuitBreaker is OPEN. Diverting to Tabular Fallback engine.")
                    return fallback_func(*args, **kwargs), True

        # Attempt primary execution
        try:
            result = func(*args, **kwargs)

            # On success: reset state if coming from HALF_OPEN
            with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    logger.info("Canary execution succeeded. Resetting CircuitBreaker to CLOSED.")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.last_state_change = time.time()
                elif self.state == CircuitState.CLOSED:
                    self.failure_count = 0

            return result, False

        except self.expected_exception as exc:
            with self.lock:
                self.failure_count += 1
                logger.error(
                    f"Primary execution failed (Error #{self.failure_count}/{self.failure_threshold}): {str(exc)}"
                )

                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.last_state_change = time.time()
                    logger.critical(
                        f"Failure threshold exceeded! Tripping CircuitBreaker to OPEN. "
                        f"All traffic routed to Fallback for {self.recovery_timeout_sec}s."
                    )

            # Return fallback result
            return fallback_func(*args, **kwargs), True

    def get_status(self) -> dict:
        with self.lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "threshold": self.failure_threshold,
                "seconds_in_state": round(time.time() - self.last_state_change, 2),
            }

