# ==========================================================
# STAGE 1: Build & Compile Python Dependencies
# ==========================================================
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install dependencies into virtualenv
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================================
# STAGE 2: Lightweight Runtime Execution Environment
# ==========================================================
FROM python:3.10-slim AS runner

WORKDIR /app

# Install curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Create non-root user first
RUN groupadd -r fraudapp && useradd -r -g fraudapp fraudapp
RUN chown -R fraudapp:fraudapp /app

# Copy virtualenv and code from builder with correct ownership
COPY --from=builder --chown=fraudapp:fraudapp /opt/venv /opt/venv
COPY --chown=fraudapp:fraudapp . .

# Switch to non-root user
USER fraudapp

# Adjust environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV ONNX_MODEL_PATH=artifacts/production/fraud_model.onnx

# Expose API serving port (FastAPI gateway)
EXPOSE 8000
# Expose Streamlit dashboard port
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the application using Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
