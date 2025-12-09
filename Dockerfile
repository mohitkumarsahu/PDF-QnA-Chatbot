FROM python:3.10-slim

WORKDIR /home/chatbot

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create data directory
RUN mkdir -p __data__

# Run the application using dynamic port from Render
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 Gen_AI_final_pdfchat:app

