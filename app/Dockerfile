FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Run the FastAPI server
CMD ["uvicorn", "main.py:app", "--host", "0.0.0.0", "--port", "8080"]