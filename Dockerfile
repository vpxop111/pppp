# Dockerfile

# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy over the requirements file to install Python dependencies
COPY requirements.txt requirements.txt

# Install dependencies using pip
RUN pip install -r requirements.txt

# Copy the entire current directory into the container's working directory
COPY . .

# Command to run the FastAPI application with Hypercorn on port 5000
CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:5000"]
