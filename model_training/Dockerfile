# Base image
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

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all required project files
COPY pipeline.py .
COPY github_repo_features_new.csv .
COPY test_with_samples.ipynb .
COPY model_training.ipynb .
COPY github_test_repos.csv .
COPY github_top_repos.csv .

CMD ["sh", "-c", "python pipeline.py && mv best_model.pkl /home/appuser/my_project/"]
