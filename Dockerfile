FROM python:3.11-slim

# Install system-level dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first (for cache efficiency)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy the entire project
COPY . .

# Expose Flask/Gunicorn port
EXPOSE 5000

# Run the server (edit app:app if your Flask instance is different)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]