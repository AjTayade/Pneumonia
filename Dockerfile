# Use a slim Python image as the base to save space and memory.
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
# Using --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 10000

# Command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
