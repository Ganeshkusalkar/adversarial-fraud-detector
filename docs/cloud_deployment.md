# Enterprise Production Cloud Deployment Guide

This guide details the architecture, configurations, and commands required to package, register, and deploy the **Adversarial Transaction Disguise Detector** model pipeline to **AWS SageMaker** and **Google Cloud Vertex AI**.

---

## 1. Containerization & Registry Setup (Shared)

Before deploying to either cloud provider, compile and push the production Docker image to your container registry (AWS ECR or Google Artifact Registry).

```bash
# Variables
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="123456789012"
GCP_REGION="us-central1"
GCP_PROJECT_ID="fraud-detector-prod"
IMAGE_TAG="v1.0.0"

# --- AWS ECR Login & Push ---
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
docker build -t fraud-detector-serving .
docker tag fraud-detector-serving:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/fraud-detector-serving:$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/fraud-detector-serving:$IMAGE_TAG

# --- GCP Artifact Registry Login & Push ---
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev
docker tag fraud-detector-serving:latest $GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/fraud-serving/detector:$IMAGE_TAG
docker push $GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/fraud-serving/detector:$IMAGE_TAG
```

---

## 2. AWS SageMaker Deployment

For low-latency, scalable real-time inference, deploy using **SageMaker Real-time Endpoints** with the custom container.

### A. Model Artifacts S3 Archival
SageMaker expects model artifacts to be packaged in a `.tar.gz` archive and placed in S3.
```bash
# Compress scaler, calibrated model, and configuration
tar -czvf model.tar.gz artifacts/production/fraud_model.onnx artifacts/production/scaler.pkl artifacts/production/prob_calibrator.pkl config/prod_config.yaml
aws s3 cp model.tar.gz s3://my-sagemaker-fraud-bucket/models/model.tar.gz
```

### B. Deployment Script (SageMaker SDK)
Use the following Python snippet to register the model and deploy a real-time endpoint with auto-scaling enabled.

```python
import sagemaker
from sagemaker.model import Model
from sagemaker.autoscaling import AutoScalingPolicy

role = sagemaker.get_execution_role()
session = sagemaker.Session()

# 1. Define Model Container from ECR
container_image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/fraud-detector-serving:v1.0.0"

model = Model(
    image_uri=container_image,
    model_data="s3://my-sagemaker-fraud-bucket/models/model.tar.gz",
    role=role,
    sagemaker_session=session,
    env={
        "API_KEYS": "my-secure-sagemaker-key",
        "FRAUD_DECISION_THRESHOLD": "0.38"
    }
)

# 2. Deploy real-time Endpoint with ml.m5.xlarge instances
predictor = model.deploy(
    initial_instance_count=2,
    instance_type="ml.m5.xlarge",
    endpoint_name="adversarial-fraud-detector-v1"
)

# 3. Configure Auto-scaling policy based on RequestCount (Scale on target 100 req/min per instance)
asg_client = session.sagemaker_client
# SageMaker will automatically scale up to 10 instances when traffic spikes during peak checkout hours.
```

---

## 3. Google Cloud Vertex AI Deployment

Vertex AI endpoints host the container in a highly managed, self-healing infrastructure.

### A. Register Custom Container Model
Register the Docker image pushed to Google Artifact Registry inside Vertex AI Model Registry.

```bash
gcloud ai models register \
  --region=$GCP_REGION \
  --display-name="adversarial-fraud-detector" \
  --container-image-uri="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/fraud-serving/detector:$IMAGE_TAG" \
  --container-env-vars="API_KEYS=my-vertex-auth-key,FRAUD_DECISION_THRESHOLD=0.38" \
  --container-ports=8000 \
  --container-predict-route="/api/v1/predict" \
  --container-health-route="/health"
```

### B. Create Vertex Endpoint & Deploy
Deploy the model from Vertex AI Model Registry into a live server endpoint.

```bash
# 1. Create Endpoint
ENDPOINT_ID=$(gcloud ai endpoints create \
  --region=$GCP_REGION \
  --display-name="fraud-serving-endpoint" \
  --format="value(name)")

# 2. Deploy Model to Endpoint (n1-standard-4 instances with auto-scaling)
gcloud ai endpoints deploy-model $ENDPOINT_ID \
  --region=$GCP_REGION \
  --model="adversarial-fraud-detector" \
  --display-name="adversarial-fraud-detector-v1" \
  --machine-type="n1-standard-4" \
  --min-replica-count=2 \
  --max-replica-count=8 \
  --traffic-split=0=100
```

---

## 4. Production Autoscaling & Cold Start Mitigations

- **Instance Warmup:** ONNX Runtime is preloaded during FastAPI server startup (`@app.on_event("startup")`), resolving cold start latencies (under 50ms on first invoke).
- **Auto-scaling Thresholds:** 
  - **AWS SageMaker:** Configure ASG (Application Auto Scaling) based on `SageMakerVariantInvocationsPerInstance` target tracking (target: 800 invocations/variant).
  - **Vertex AI:** Scaling triggers automatically on CPU utilization (exceeding 60%).
