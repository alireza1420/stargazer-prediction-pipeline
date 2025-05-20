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
COPY "data_preprocessing_plus_model_training.ipynb" .
COPY "test_with_a_sample.ipynb" .
COPY "github_test_repos.csv" .

EXPOSE 8888

# Run the model script
CMD ["sh", "-c", "jupyter nbconvert --to notebook --execute --inplace data_preprocessing_plus_model_training.ipynb && jupyter nbconvert --to notebook --execute --inplace test_with_a_sample.ipynb"]

