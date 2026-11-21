from prometheus_client import Counter, Histogram, Gauge

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
    buckets=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
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

# Real-time subgraph topology size metrics
SUBGRAPH_NODES_EVALUATED = Histogram(
    'subgraph_nodes_evaluated',
    'Number of graph nodes extracted in the localized transaction neighborhood',
    buckets=[1, 2, 3, 5, 10, 20, 50]
)

SUBGRAPH_EDGES_EVALUATED = Histogram(
    'subgraph_edges_evaluated',
    'Number of graph edges evaluated in the localized transaction neighborhood',
    buckets=[0, 1, 2, 4, 8, 16, 32, 64]
)

# Circuit breaker state: 0 = CLOSED (normal), 1 = HALF_OPEN (testing), 2 = OPEN (tripped)
CIRCUIT_BREAKER_STATE = Gauge(
    'circuit_breaker_state',
    'Current state of the serving circuit breaker (0: CLOSED, 1: HALF_OPEN, 2: OPEN)'
)

# Fallback predictions counter
FALLBACK_PREDICTIONS_TOTAL = Counter(
    'fallback_predictions_total',
    'Total predictions served using the fast tabular fallback engine',
    ['reason']
)
