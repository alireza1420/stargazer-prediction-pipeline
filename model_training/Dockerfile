# Base image
FROM python:3.10-slim

# Set working directory early
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY test_with_samples.ipynb .
COPY model_training.ipynb .
COPY pipeline.py .
COPY github_repo_features_new.csv .
COPY github_test_repos.csv .
COPY github_top_repos.csv .

EXPOSE 8888

# Run both notebooks
CMD ["sh", "-c", "jupyter nbconvert --to notebook --execute --inplace model_training.ipynb && jupyter nbconvert --to notebook --execute --inplace test_with_samples.ipynb && mv best_model.pkl /home/appuser/project_results/"]
