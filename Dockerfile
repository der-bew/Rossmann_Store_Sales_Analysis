# Set the base image to Python 3.11
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run the command to run the application
CMD ["python", "src/rossmann_store_analysis/__main__.py"]