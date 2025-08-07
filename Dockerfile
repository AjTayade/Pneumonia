# Use the latest stable TensorFlow CPU image as the base.
# This image comes with Python and TensorFlow pre-installed.
FROM tensorflow/tensorflow:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the smaller, remaining dependencies, ignoring packages already installed in the base image
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 10000

# Command to run the application using Gunicorn (a production-ready server)
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
