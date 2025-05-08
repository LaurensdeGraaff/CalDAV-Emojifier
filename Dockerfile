FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and configuration files
# COPY src/ ./src/
COPY config/ ./config/
COPY sync.py ./sync.py
COPY entrypoint.sh /app/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set environment variables for configuration

ARG LOG_LEVEL

ENV SYNC_INTERVAL=3600
ENV LOG_LEVEL=$LOG_LEVEL
ENV MASTER_ON_COLLISION=$MASTER_ON_COLLISION

# Set the entrypoint to the script
ENTRYPOINT ["/app/entrypoint.sh"]