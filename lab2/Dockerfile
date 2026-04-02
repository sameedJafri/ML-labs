# ── Build stage: install deps ─────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Install CPU-only PyTorch (avoids ~2 GB of CUDA libraries)
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining serving dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    Flask==3.1.3 \
    waitress==3.0.2 \
    python-dotenv==1.0.1 \
    "Pillow>=10.0.0" \
    "numpy>=1.24.0"

# Copy weights and application code
# weights/ must exist locally before building (run train.py first)
COPY weights/ weights/
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
