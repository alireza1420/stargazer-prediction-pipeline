# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app
RUN pip install -r requirements.txt

# Copy source files
COPY data data preprocessing_plus_model training.ipynb .
COPY test with a sample.ipynb .
COPY github_test_repos.csv .


EXPOSE 8888

# Run the model script
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
