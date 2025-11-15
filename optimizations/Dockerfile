FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY data_ingestion_src/s3_client.py ./
COPY video_processor.py ./
COPY inference_pipeline.py ./

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "inference_pipeline.py", "--help"]