from prometheus_client import Counter, Histogram

# Total fraud predictions counter
FRAUD_PREDICTIONS_TOTAL = Counter(
    'fraud_predictions_total',
    'Total number of fraud predictions made',
    ['flagged']
)

# Inference latency histogram
INFERENCE_LATENCY_SECONDS = Histogram(
    'inference_latency_seconds',
    'Latency of fraud predictions in seconds',
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
)

# Raw fraud score distribution
FRAUD_SCORE_DISTRIBUTION = Histogram(
    'fraud_score_distribution',
    'Distribution of raw fraud probability scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Model errors counter
MODEL_ERRORS_TOTAL = Counter(
    'model_errors_total',
    'Total number of inference or validation errors',
    ['error_type']
)
