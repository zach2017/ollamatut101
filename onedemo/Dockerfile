# Use a slim Python image for efficiency
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .
COPY one.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables (optional, but helpful for debugging)
ENV PYTHONUNBUFFERED=1

# Run the script when the container starts
CMD ["python", "one.py"]