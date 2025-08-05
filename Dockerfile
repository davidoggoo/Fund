# Multi-stage Docker build for RTAI Trading System
# Target: <190MB production image with zero matplotlib dependencies

# Stage 1: Builder - Install dependencies and compile bytecode
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install Python dependencies to /app/venv
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install dependencies (excluding matplotlib/mplfinance - banned)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Minimal production image
FROM python:3.11-slim as runtime

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash rtai

# Copy Python virtual environment from builder
COPY --from=builder /app/venv /app/venv

# Copy application code
WORKDIR /app
COPY --chown=rtai:rtai . .

# Set environment
ENV PATH="/app/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RTAI_LOG_LEVEL=INFO \
    RTAI_ENVIRONMENT=production

# Create required directories
RUN mkdir -p logs output questdb && \
    chown -R rtai:rtai /app

# Switch to non-root user
USER rtai

# Compile Python bytecode for faster startup
RUN python -m compileall -b rtai/

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8080

# Default command - FastAPI server
CMD ["uvicorn", "rtai.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
