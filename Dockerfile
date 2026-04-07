# FrontierLabs-Env Dockerfile
# Hugging Face Spaces compatible (port 7860, non-root user)

FROM python:3.11
# HF Spaces metadata
LABEL maintainer="FrontierLabs Team"
LABEL org.opencontainers.image.title="FrontierLabs-Env"
LABEL org.opencontainers.image.description="OpenEnv AI Infrastructure Simulation Sandbox"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY openenv.yaml .
COPY environment.py .
COPY graders.py .
COPY main.py .
COPY inference.py .

# Fix permissions
RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]