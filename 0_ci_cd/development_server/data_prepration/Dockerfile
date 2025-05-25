# Use a slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (best practice for Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY data_preparation.py .
COPY github_repo_features_new.csv .
COPY github_top_repos_raw.json .

# Run training script to generate best_model.pkl
CMD ["python", "data_preparation.py"]
