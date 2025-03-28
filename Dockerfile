# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && pip install --no-cache-dir --timeout=120 --retries=2 -r requirements.txt
    
# Set the WANDB_API_KEY environment variable #dont do this in production
#ENV WANDB_API_KEY=your_api_key

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "main.py"]
