# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && pip install --no-cache-dir --timeout=120 --retries=2 -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World
   
# Set environment variables to control CUDA behavior
ENV CUDA_CACHE_DISABLE=1
ENV CUDA_LAUNCH_BLOCKING=1
ENV CUDA_VISIBLE_DEVICES=0

# Run app.py when the container launches
CMD ["python", "main.py"]
