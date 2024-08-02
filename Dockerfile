# Stage 1: Build dependencies
FROM python:3.9-alpine as builder

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Build application
FROM python:3.9-alpine

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Copy dependencies from stage 1
COPY --from=builder /app/. .

# Run command
CMD ["python", "generate_ai_sentences.py"]
