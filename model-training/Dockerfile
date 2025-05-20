# Base image
FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY "requirements.txt" .
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app

# Copy source files
COPY "test_with_samples.ipynb" .
COPY "model_training.ipynb" .
COPY "pipeline.py" .
COPY "github_repo_features_new.csv"
COPY "github_test_repos.csv" .
COPY "github_top_repos.csv" .

EXPOSE 8888

# Run the model script
CMD ["sh", "-c", "jupyter nbconvert --to notebook --execute --inplace model_training.ipynb && jupyter nbconvert --to notebook --execute --inplace test_with_samples.ipynb"]

