# ai_backend/Dockerfile
FROM python:3.10-slim AS development

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements from ai_backend (since build context = ./ai_backend)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

COPY . .

CMD ["python", "main.py"]
