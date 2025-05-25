FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p models data

# Set environment variables
ENV PYTHONPATH=/app
ENV MODELS_DIR=/app/models
ENV DATA_DIR=/app/data

# Expose the API port
EXPOSE 8000

# Command to run the API
CMD ["python", "run.py"]
