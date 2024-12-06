# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Earth Engine
RUN pip install --no-cache-dir \
    earthengine-api \
    earthaccess \
    geemap \
    pytorch-lightning==1.9.5

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads \
    templates/assets/results \
    data \
    methane-models

# Set up environment variables (these should be overridden at runtime)
ENV EARTHDATA_USERNAME="" \
    EARTHDATA_PASSWORD="" \
    OPENAI_API_KEY="" \
    NIXTLA_API_KEY="" \
    FLASK_APP=app.py \
    FLASK_ENV=production

# Expose port
EXPOSE 5000

# Create a non-root user
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/status || exit 1

# Command to run the application
CMD ["python", "app.py"]
