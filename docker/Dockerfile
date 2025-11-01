FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev

# Copy application code
COPY graphrag/ ./graphrag/

# Create data directories
RUN mkdir -p /app/data/{traces,faiss_index}

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "graphrag.app.api.routers:app", "--host", "0.0.0.0", "--port", "8000"]

