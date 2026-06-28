import time
import logging
from collections import deque

logger = logging.getLogger("AlertingEngine")

class AlertingEngine:
    def __init__(self, window_seconds=300):
        self.window_seconds = window_seconds
        self.predictions = deque()
        self.latencies = deque()
        self.errors = deque()

    def record_prediction(self, is_fraud: bool):
        now = time.time()
        self.predictions.append((now, is_fraud))
        self._prune(now)
        self.check_fraud_rate()

    def record_latency(self, latency_ms: float):
        now = time.time()
        self.latencies.append((now, latency_ms))
        self._prune(now)
        self.check_latency()

    def record_error(self):
        now = time.time()
        self.errors.append(now)
        self._prune(now)
        self.check_error_rate()

    def _prune(self, now):
        cutoff = now - self.window_seconds
        while self.predictions and self.predictions[0][0] < cutoff:
            self.predictions.popleft()
        while self.latencies and self.latencies[0][0] < cutoff:
            self.latencies.popleft()
        while self.errors and self.errors[0] < cutoff:
            self.errors.popleft()

    def check_fraud_rate(self):
        if len(self.predictions) < 100:
            return
        fraud_count = sum(1 for _, is_fraud in self.predictions if is_fraud)
        fraud_rate = fraud_count / len(self.predictions)
        if fraud_rate > 0.15:
            logger.critical(f"ALERT: High fraud rate detected! {fraud_rate*100:.1f}% over the last 5 minutes.")

    def check_latency(self):
        if len(self.latencies) < 100:
            return
        sorted_latencies = sorted(l for _, l in self.latencies)
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        if p99 > 200:
            logger.critical(f"ALERT: High latency detected! P99 is {p99:.1f}ms over the last 5 minutes.")

    def check_error_rate(self):
        total_requests = len(self.predictions) + len(self.errors)
        if total_requests < 100:
            return
        error_rate = len(self.errors) / total_requests
        if error_rate > 0.01:
            logger.critical(f"ALERT: High error rate detected! {error_rate*100:.1f}% over the last 5 minutes.")

alerting_engine = AlertingEngine()
