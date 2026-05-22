FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file
COPY . .

# Download artifacts
RUN python download_artifacts.py

# Hugging Face pakai port 7860
EXPOSE 7860

CMD ["python", "main_hf.py"]
