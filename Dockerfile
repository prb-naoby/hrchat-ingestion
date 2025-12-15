# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1

# System deps untuk Docling/Pillow/Tesseract
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  build-essential \
  poppler-utils \
  tesseract-ocr \
  libjpeg-dev zlib1g-dev libpng-dev libstdc++6 \
  curl ca-certificates && \
  rm -rf /var/lib/apt/lists/*

# Buat user non-root
RUN useradd -m -u 10001 appuser
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY src/ ./src
RUN chown -R appuser:appuser /app
USER appuser

# Jalankan API — bind ke 0.0.0.0 di DALAM container (tidak dipublish keluar)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
