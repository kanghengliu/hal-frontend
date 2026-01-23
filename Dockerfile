FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy app code
COPY . .

EXPOSE 8742

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8742", "--workers", "4"]
