# Use Python 3.10
FROM python:3.10-slim

# System deps for OpenCV / MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# App directory
WORKDIR /app

# Install Python deps + gunicorn
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy the rest of the project
COPY . .

# We will just listen on 8000 inside the container
ENV PORT=8000
EXPOSE 8000

# Exec-form CMD (no shell, no $PORT interpolation needed)
CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:8000"]
