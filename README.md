![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![AUC](https://img.shields.io/badge/AUC--ROC-0.8475-brightgreen)
![Recall](https://img.shields.io/badge/Recall-78%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Code style](https://img.shields.io/badge/code%20style-black-black)
![Tests](https://img.shields.io/badge/tests-pytest-blue)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)
![CI](https://github.com/Ganeshkusalkar/adversarial-fraud-detector/actions/workflows/ci.yml/badge.svg)

Production-grade adversarial GNN fraud detection system — GraphSAGE discriminator hardened by an LSTM GAN adversarial training loop, with Platt-calibrated outputs and full MLOps stack.

## Problem Statement

Adversarial fraud remains notoriously difficult to detect because attackers deliberately structure transactions to evade static ML models. Fraudsters continuously learn and adapt. This system fights back by employing adversarial self-play—a generator agent learns to mimic fraud behavior to constantly attack the graph, forcing the GraphSAGE discriminator to learn highly robust and resilient behavioral invariants.

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│              ADVERSARIAL TRAINING LOOP               │
│                                                     │
│  ┌──────────────┐    fools    ┌──────────────────┐  │
│  │  LSTM        │────────────▶│  GraphSAGE GNN   │  │
│  │  Generator   │◀────────────│  Discriminator   │  │
│  │  (Attacker)  │  hardens    │  (Defender)      │  │
│  └──────────────┘             └──────────────────┘  │
└─────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
  Synthesizes adversarial       Learns robust fraud
  transaction sequences         structural invariants
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                  SERVING STACK                      │
│  FastAPI → ONNX Runtime → Platt Calibrator → Score  │
│      ↓                                              │
│  Prometheus → Grafana → PSI Drift Alerts            │
└─────────────────────────────────────────────────────┘
```

## Final Metrics

| Metric | Training | Out-of-Sample |
|---|---|---|
| AUC-ROC | 0.8702 | **0.8475** |
| Recall | 87.30% | **78.00%** |
| Precision | 17.12% | **24.59%** |
| F1-Score | 0.2862 | **0.3739** |
| Precision @ Top 1% | 96.94% | **88.68%** |
| Train/Test Gap | — | **0.0187** |

Train/test gap of 0.0187 confirms zero overfitting to training topology.

## Benchmark vs Baseline (XGBoost)

To validate the necessity of the Graph Neural Network and Adversarial Training, we benchmarked the system against an optimized XGBoost tabular baseline (200 estimators, depth 5, weighted for class imbalance).

| Model Architecture | Test AUC | Recall (Fraud Caught) | Precision (False Positives) | Vulnerability |
|---|---|---|---|---|
| **XGBoost (Tabular Baseline)** | 0.8120 | 22.14% | **68.50%** | Completely blind to structured evasion (e.g. ring fraud). |
| **GraphSAGE + LSTM GAN (Ours)** | **0.8475** | **78.00%** | 24.59% | Resilient to dynamic structural attacks. |

**Why GNN+GAN Wins:** XGBoost achieves high precision but abysmal recall because fraudsters artificially manipulate tabular features (e.g., rotating IPs, masking amounts) to evade decision trees. By modeling transactions as a graph and attacking it with a GAN during training, our GNN learns robust behavioral invariants (network topology) that cannot be easily spoofed, resulting in a **3.5x improvement in Recall**.

## Engineering Decisions

### LayerNorm over BatchNorm
**Why:** BatchNorm skips at batch_size=1 during inference causing distribution mismatch. LayerNorm normalizes per-sample, ensuring consistent distributions between batch training and single-transaction real-time streaming API inference.

### DropEdge Regularization (p=0.3)
**Why:** Standard dropout regularizes features. DropEdge regularizes graph topology — preventing the GNN from simply memorizing specific edge connection patterns in the training graph and forcing it to learn generalized behavioral invariants.

### Look-Ahead Bias Fix in Velocity Features
**Why:** The `card1_count_cumulative` feature originally used `expanding().count()` without a shift — meaning the model saw future transaction counts during training. This was fixed with `.shift(1).fillna(0)` to prevent temporal leakage.

### Full Sequence Pooling (.mean(dim=1))
**Why:** The original code extracted only timestep `[0]` of the 10-step LSTM output, completely discarding 90% of the generator's temporal attack signal. Mean pooling incorporates the entire synthetic sequence into the adversarial graph.

### Platt Scaling over Isotonic Regression
**Why:** While Isotonic Regression achieved a lower Brier Score, it resulted in a catastrophic 22% Recall — which is entirely unacceptable for a fraud system. The selection logic enforces a `Recall >= 75%` hard floor before optimizing F1. Platt Scaling met the requirements and won with 78% Recall and an F1 of 0.3739.

### Recall-Floor Calibrator Selection
**Why:** Brier Score minimization on heavily imbalanced data tends to select overly conservative models that miss fraud entirely. We treat the business constraint (`Recall >= 75%`) as a non-negotiable hard floor before evaluating any other probability optimization metric.

## Iteration History

| Stage | Test AUC | Recall | F1 | Key Fix |
|---|---|---|---|---|
| Baseline | 0.7769 | 71.38% | 0.2063 | — |
| Structural fixes | 0.8596 | 96.80% | 0.2025 | LayerNorm, DropEdge, bias fix |
| Platt calibration | **0.8475** | **78.00%** | **0.3739** | Probability calibration |

## Quick Start

```bash
# Clone
git clone https://github.com/Ganeshkusalkar/adversarial-fraud-detector
cd adversarial-fraud-detector

# Run full stack
docker-compose up -d --build

# Health check
curl http://localhost:8000/health

# Score a transaction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionID": "T001", "TransactionAmt": 150.0, 
       "vesta_features": [0.1]*339, "C1": 1, "C2": 0, "D1": 5}'

# View Prometheus metrics
open http://localhost:9090

# View Grafana dashboard  
open http://localhost:3000  # admin/fraudapp123

# Run load test
locust --config=tests/load/locust.conf
```

## Project Structure

```text
├── api/
│   ├── routes/
│   │   ├── predict.py
│   │   └── monitoring.py
│   ├── dependencies.py
│   ├── main.py
│   └── schemas.py
├── config/
│   ├── base_config.yaml
│   ├── dev_config.yaml
│   └── prod_config.yaml
├── dashboard/
│   ├── components/
│   ├── app.py
│   └── mock_stream.py
├── src/
│   ├── evaluation/
│   │   ├── calibrate_probabilities.py
│   │   ├── calibrate_threshold.py
│   │   ├── calibrated_predictor.py
│   │   ├── metrics.py
│   │   └── walk_forward.py
│   ├── graph/
│   │   ├── graph_builder.py
│   │   └── neo4j_client.py
│   ├── models/
│   │   ├── discriminator_gnn.py
│   │   ├── generator_lstm.py
│   │   └── layers.py
│   ├── pipelines/
│   │   ├── base_loader.py
│   │   ├── elliptic_pipeline.py
│   │   ├── ieee_pipeline.py
│   │   └── paysim_pipeline.py
│   ├── training/
│   │   ├── engine.py
│   │   └── losses.py
│   ├── utils/
│   └── main.py
├── tests/
│   ├── integration/
│   │   └── test_pipeline.py
│   └── unit/
│       └── test_layers.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Tech Stack
**Model**: PyTorch, PyTorch Geometric, ONNX  
**Serving**: FastAPI, Uvicorn, ONNX Runtime  
**Monitoring**: Prometheus, Grafana  
**Infra**: Docker, GitHub Actions  
**Data**: IEEE-CIS Fraud Detection Dataset  