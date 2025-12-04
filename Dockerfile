# Use Python 3.10 (mediapipe + opencv-contrib are happiest here)
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for OpenCV, mediapipe, video, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

# (Optional but recommended) avoid NumPy 2.x ABI issues:
# If you hit NumPy-related crashes, change your requirements line to:
# numpy==1.26.4
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Copy the project (templates, static, data, etc.)
COPY . .

# Railway injects PORT; default to 8000 locally
ENV PORT=8000

EXPOSE 8000

# app.py contains `app = Flask(__name__)`, so we use app:app
CMD ["sh", "-c", "gunicorn app:app --workers 2 --timeout 180 --bind 0.0.0.0:${PORT}"]
