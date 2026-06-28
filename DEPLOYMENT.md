# Production Deployment Guide

## Prerequisites
- Docker & Docker Compose
- 2 CPU Cores, 2GB RAM minimum

## Startup
To start the hardened infrastructure locally:
```bash
docker-compose up -d --build
```
This will start:
- FastAPI Gateway (Port 8000)
- Streamlit Dashboard (Port 8501)
- Prometheus Metrics (Port 9090)
- Grafana Dashboard (Port 3000)
- Neo4j Database (Port 7474 & 7687)

## Load Testing
Ensure Locust is installed, then run:
```bash
locust --config=tests/load/locust.conf
```
Target: 0% failure rate, P99 latency < 200ms.

## Model Registry CLI
You can promote and rollback models via the CLI:
```bash
# Promote model version 1 to production
python -m src.utils.model_registry promote v1

# Rollback to version 1
python -m src.utils.model_registry rollback v1 --reason "High latency"
```
