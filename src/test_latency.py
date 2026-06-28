"""
src/test_latency.py
===================
Runs a formal latency benchmark on the production ONNX model.
Proves that the model meets the < 100ms P99 latency requirement
for real-time transaction processing.
"""

import time
import numpy as np
import onnxruntime as ort
import yaml

from src.utils.logger import setup_logger

logger = setup_logger("LatencyBenchmark")

def run_latency_test():
    model_path = "artifacts/production/fraud_model.onnx"
    config_path = "config/base_config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    in_channels = config["model"]["gnn"]["in_channels"]
    
    logger.info("=" * 60)
    logger.info("  PRODUCTION INFERENCE LATENCY BENCHMARK (ONNX)")
    logger.info("=" * 60)
    
    try:
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        # Configure for single-threaded CPU inference to simulate basic cloud container
        sess_options.intra_op_num_threads = 1
        
        session = ort.InferenceSession(model_path, sess_options=sess_options)
        input_name = session.get_inputs()[0].name
        edge_name  = session.get_inputs()[1].name
    except Exception as e:
        logger.error(f"Failed to load ONNX model. Run training first. {e}")
        return

    n_iterations = 1000
    latencies = []
    
    logger.info(f"Warming up ONNX session...")
    for _ in range(10):
        dummy_x = np.random.randn(1, in_channels).astype(np.float32)
        dummy_edges = np.zeros((2, 1), dtype=np.int64)
        session.run(None, {input_name: dummy_x, edge_name: dummy_edges})
        
    logger.info(f"Running {n_iterations} single-node inferences...")
    for _ in range(n_iterations):
        # Simulate a single real-time transaction arriving
        dummy_x = np.random.randn(1, in_channels).astype(np.float32)
        dummy_edges = np.zeros((2, 1), dtype=np.int64)
        
        start_time = time.perf_counter()
        _ = session.run(None, {input_name: dummy_x, edge_name: dummy_edges})
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000) # Convert to ms
        
    latencies = np.array(latencies)
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    max_lat = np.max(latencies)
    
    print()
    print(f"Latency Results (over {n_iterations} runs):")
    print("-" * 40)
    print(f"P50 (Median) : {p50:.2f} ms")
    print(f"P95          : {p95:.2f} ms")
    print(f"P99          : {p99:.2f} ms")
    print(f"Max          : {max_lat:.2f} ms")
    print("-" * 40)
    
    if p99 < 100.0:
        print("[PASS] STATUS: PASS - Model meets production SLA (< 100ms P99)")
    else:
        print("[FAIL] STATUS: FAIL - Model exceeds production SLA")

if __name__ == "__main__":
    run_latency_test()
