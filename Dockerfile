# ==========================================================
# STAGE 1: Build & Compile Python Dependencies
# ==========================================================
FROM python:3.11.9-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# ==========================================================
# STAGE 2: Lightweight Runtime Execution Environment
# ==========================================================
FROM python:3.11.9-slim AS runner

RUN useradd -m -r fraudapp && \
    apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /root/.local /home/fraudapp/.local
COPY . .

RUN chown -R fraudapp:fraudapp /app

USER fraudapp
ENV PATH=/home/fraudapp/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
