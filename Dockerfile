# Deepfake Detection via Frequency Analysis & Schrödinger Bridges
# Supports both CPU and GPU (NVIDIA)

FROM python:3.11-slim

# Build argument for GPU support
ARG USE_GPU=false

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch (CPU or GPU based on build arg)
RUN if [ "$USE_GPU" = "true" ]; then \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install remaining Python dependencies (skip torch as it's already installed)
RUN grep -v "^torch" requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Copy project files
COPY core/ ./core/
COPY main.py analyze_data.py test_localized.py ./

# Create necessary directories
RUN mkdir -p \
    data/cnn \
    data/dct \
    data/localized \
    data/test \
    experiments/checkpoints \
    artifacts/default

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command - show help
CMD ["python", "main.py", "--help"]
