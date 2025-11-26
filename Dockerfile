# Dockerfile for LockerLink AI microservice
# Base image with CUDA 12.6+ support for SAM3
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Install Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Note: PyTorch with CUDA 12.6 support
RUN pip3.12 install --no-cache-dir torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip3.12 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Copy SAM3 directory (should be cloned before building)
# If SAM3 is not present, the build will fail - user must clone it first
COPY sam3/ ./sam3/

# Install SAM3 in editable mode
RUN if [ -d "./sam3" ] && [ -f "./sam3/setup.py" ]; then \
        pip3.12 install --no-cache-dir -e ./sam3; \
    else \
        echo "WARNING: SAM3 directory not found. Please clone SAM3 repo before building Docker image."; \
        exit 1; \
    fi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3.12 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

