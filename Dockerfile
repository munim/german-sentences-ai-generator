# Stage 1: Build dependencies
FROM python:3.9-alpine as builder

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run command
CMD ["python", "./gen_ai_sentences.py"]
