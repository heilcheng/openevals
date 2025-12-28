# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install -r requirements.txt

# Install flash attention (optional, might fail on some systems)
RUN pip install flash-attn --no-build-isolation || echo "Flash attention installation failed, continuing without it"

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create directories for data and results
RUN mkdir -p /app/data /app/results /app/cache

# Set up environment for HuggingFace
ENV HF_HOME=/app/cache/huggingface \
    TRANSFORMERS_CACHE=/app/cache/transformers \
    HF_DATASETS_CACHE=/app/cache/datasets

# Create non-root user for security
RUN useradd -m -u 1000 gemma && \
    chown -R gemma:gemma /app

USER gemma

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import openevals; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "openevals.scripts.run_benchmark", "--help"]

# Labels for metadata
LABEL maintainer="Hailey Cheng <hailey.cheng@example.com>" \
      description="OpenEvalsing Suite - Production-ready evaluation framework" \
      version="1.0.0" \
      org.opencontainers.image.source="https://github.com/heilcheng/gemma-benchmark"
