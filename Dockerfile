# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    openjdk-11-jdk \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Compile C++ module
RUN cd src/cpp && ./compile.bat

# Expose port for web application
EXPOSE 8000

# Run the application
CMD ["gunicorn", "--worker-class", "uvicorn.workers.UvicornWorker", "src.web.app:app"]
