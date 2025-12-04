# Use Python 3.10
FROM python:3.10-slim

# System deps for OpenCV / MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# App directory
WORKDIR /app

# Install Python deps (and gunicorn explicitly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy the rest of the project
COPY . .

# Railway will set PORT, but keep a fallback for local runs
ENV PORT=8000

EXPOSE 8000

# Shell-form so ${PORT} is expanded by the shell
CMD gunicorn app:app --bind "0.0.0.0:${PORT}"
