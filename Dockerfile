# Multi-stage Docker build for AI Drug Discovery Platform

# Stage 1: Base image with RDKit
FROM continuumio/miniconda3:latest as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with RDKit
RUN conda create -n drugdiscovery python=3.11 -y && \
    conda install -n drugdiscovery -c conda-forge rdkit -y

# Stage 2: Application
FROM base

WORKDIR /app

# Activate conda environment in shell
SHELL ["conda", "run", "-n", "drugdiscovery", "/bin/bash", "-c"]

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p results data/processed logs

# Expose ports
EXPOSE 8000 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["conda", "run", "--no-capture-output", "-n", "drugdiscovery", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
