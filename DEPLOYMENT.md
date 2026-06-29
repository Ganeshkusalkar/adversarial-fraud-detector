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

## Cloud Deployment (AWS / GCP)

This stack is designed to be easily deployed to managed container services.

### AWS ECS (Fargate)
1. **Push to ECR:**
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
   docker build -t fraud-detector .
   docker tag fraud-detector:latest <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/fraud-detector:latest
   docker push <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/fraud-detector:latest
   ```
2. **Deploy via ECS:**
   - Create an ECS Cluster (Fargate type).
   - Create a Task Definition utilizing the uploaded ECR image.
   - Attach an Application Load Balancer (ALB) pointing to container port 8000.
   - Configure Auto-scaling based on CPU utilization > 70%.

### GCP Cloud Run
1. **Build and Submit:**
   ```bash
   gcloud builds submit --tag gcr.io/<PROJECT_ID>/fraud-detector
   ```
2. **Deploy:**
   ```bash
   gcloud run deploy fraud-api \
       --image gcr.io/<PROJECT_ID>/fraud-detector \
       --platform managed \
       --region us-central1 \
       --allow-unauthenticated \
       --port 8000 \
       --memory 2Gi \
       --cpu 2
   ```

### Render (Free Tier Friendly)
Render is an excellent option for hosting the dashboard and API for free during portfolio reviews.
1. Create a new **Web Service** on [Render.com](https://render.com).
2. Connect your GitHub repository.
3. Choose **Docker** as the environment.
4. Render will automatically detect the `Dockerfile` and build the container.
5. In the settings, expose port `8000` for the API or `8501` for the Streamlit dashboard.

### Heroku (Low Cost)
1. Install the Heroku CLI and login:
   ```bash
   heroku login
   heroku container:login
   ```
2. Create app and push container:
   ```bash
   heroku create fraud-detector-app
   heroku container:push web -a fraud-detector-app
   heroku container:release web -a fraud-detector-app
   ```
