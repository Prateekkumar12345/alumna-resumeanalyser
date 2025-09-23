# Use the official lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install the dependencies. We use --no-cache-dir to prevent caching, which reduces image size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the working directory
COPY . .

# RUN python create_tables.py

# Expose port 8000, which Uvicorn will listen on
EXPOSE 8000

# Command to run the application.
# The --reload flag is useful for development but should be removed in production
# to improve performance and stability.
# The --host 0.0.0.0 makes the server accessible from outside the container.
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]
