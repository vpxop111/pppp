# Use the official Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Create a new user and set permissions
RUN useradd -m abhishek && chown -R abhishek:abhishek /app

# Switch to the new user
USER abhishek

# Command to run the application
CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:8000"]