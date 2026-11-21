import time
import pytest
from src.utils.circuit_breaker import CircuitBreaker, CircuitState


def test_circuit_breaker_normal_closed_operation():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_sec=1.0)

    def success_func(val):
        return val * 2

    def fallback_func(val):
        return 0

    res, fallback_used = cb.call(success_func, fallback_func, 5)
    assert res == 10
    assert fallback_used is False
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_trips_to_open_on_failures():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_sec=0.2)

    def failing_func():
        raise RuntimeError("GNN engine timeout")

    def fallback_func():
        return 0.05

    # Failure 1
    res1, fb1 = cb.call(failing_func, fallback_func)
    assert res1 == 0.05
    assert fb1 is True
    assert cb.state == CircuitState.CLOSED

    # Failure 2 (Trips)
    res2, fb2 = cb.call(failing_func, fallback_func)
    assert res2 == 0.05
    assert fb2 is True
    assert cb.state == CircuitState.OPEN


def test_circuit_breaker_recovers_to_closed():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=0.1)

    def failing_func():
        raise RuntimeError("Engine failure")

    def success_func():
        return 42

    def fallback_func():
        return -1

    # Trip to OPEN
    cb.call(failing_func, fallback_func)
    assert cb.state == CircuitState.OPEN

    # Wait for recovery timeout
    time.sleep(0.15)

    # Canary execution in HALF_OPEN recovers to CLOSED
    res, fb = cb.call(success_func, fallback_func)
    assert res == 42
    assert fb is False
    assert cb.state == CircuitState.CLOSED

