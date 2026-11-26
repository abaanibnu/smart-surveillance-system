FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=10000

# Install system dependencies for OpenCV and video handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight CPU-only PyTorch first (180MB instead of 2.5GB CUDA)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure storage directories exist
RUN mkdir -p static/frames static/annotated static/person_crops database uploads

EXPOSE 10000 7860 5000

CMD ["python", "app.py"]
