# Base image
FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

# Copy source files
COPY "data_preprocessing_plus_model_training.ipynb" .
COPY "test_with_a_sample.ipynb" .
COPY "github_test_repos.csv" .

EXPOSE 8888

# Run the model script
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
