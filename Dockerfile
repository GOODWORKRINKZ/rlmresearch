FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better caching
COPY pyproject.toml .
COPY src/ src/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV RLM_REMOTE=true
ENV RLM_WORKSPACE=/workspace

# Create workspace directory (will be mounted as volume)
RUN mkdir -p /workspace

# Expose port 8000
EXPOSE 8000

# Run the server
CMD ["python", "-m", "uvicorn", "rlm_assistant.server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
