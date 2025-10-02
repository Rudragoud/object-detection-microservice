
# # Multi-stage build for UI Backend
# FROM python:3.11-slim AS base

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     PIP_DISABLE_PIP_VERSION_CHECK=1 \
#     PIP_NO_CACHE_DIR=1

# # Create non-root user for security
# RUN groupadd -g 1000 appuser && \
#     useradd -r -u 1000 -g appuser appuser

# # Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ca-certificates \
#     curl \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

# WORKDIR /app

# # ================================
# # Development Stage
# # ================================
# FROM base AS development

# # Install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt && \
#     pip cache purge

# # Create static directories and set permissions
# RUN mkdir -p static/uploads templates static/css static/js && \
#     chown -R appuser:appuser /app

# # Switch to non-root user
# USER appuser

# EXPOSE 8000

# # Development command with reload
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# # ================================
# # Production Stage
# # ================================
# FROM base AS production

# # Copy requirements and install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt && \
#     pip cache purge

# # Copy application code with proper ownership
# COPY --chown=appuser:appuser . .

# # Create directories and set permissions
# RUN mkdir -p static/uploads && \
#     chown -R appuser:appuser /app

# # Switch to non-root user for security
# USER appuser

# # Health check for production monitoring
# HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
#     CMD curl -f http://localhost:8000/ || exit 1

# EXPOSE 8000

# # Production command with multiple workers
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]



# ui_backend/Dockerfile
FROM python:3.10-slim AS development

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

COPY . .

CMD ["python", "main.py"]
