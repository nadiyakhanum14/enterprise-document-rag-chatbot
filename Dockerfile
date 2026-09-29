# Enterprise Document RAG Chatbot — Dockerfile
#
# Build:  docker build -t enterprise-rag .
# Run:    docker run -p 8501:8501 --env-file .env -v $(pwd)/docs:/app/docs -v $(pwd)/chroma_db:/app/chroma_db enterprise-rag

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by some Python packages, OCR, and PDF rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p docs chroma_db evaluation

# Expose ports: 8501 (Streamlit UI), 8000 (FastAPI Backend)
EXPOSE 8501 8000

# Streamlit configuration to run correctly inside Docker
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Default entry point: run the Streamlit app
CMD ["streamlit", "run", "app.py"]
