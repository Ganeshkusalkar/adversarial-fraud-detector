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

# By default CI builds should avoid installing very large ML packages (torch, torch-geometric)
# Use the build-arg INSTALL_HEAVY=true to install the full requirements (for local/dev builds).
ARG INSTALL_HEAVY="false"

# Include both full and runtime requirements in the image build context
COPY requirements.txt .
COPY requirements-runtime.txt .

# Install dependencies into virtualenv (choose runtime or full list based on build arg)
RUN if [ "$INSTALL_HEAVY" = "true" ]; then \
      pip install --no-cache-dir -r requirements.txt ; \
    else \
      pip install --no-cache-dir -r requirements-runtime.txt ; \
    fi

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
