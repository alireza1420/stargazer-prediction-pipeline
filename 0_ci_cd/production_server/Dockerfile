FROM python:3.9-buster

WORKDIR /app/
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .

# Install Gunicorn
RUN pip install gunicorn

# Command to run Gunicorn (4 workers, binding to 0.0.0.0:5100)
CMD ["gunicorn", "--bind", "0.0.0.0:5100", "--workers", "4", "app:app"]