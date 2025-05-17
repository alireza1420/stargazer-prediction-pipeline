# Base image
FROM pyton:3.10-slim

# Set working directory
WORKDIR /app

# Copy source files

COPY knn.py .
COPY github_repo_features.csv .

# Install dependencies
RUN pip install --no-cache-dir numpy pandas scikit-learn pulsar-client joblib

# Run the model script
CMD ["python", "knn.py"]
