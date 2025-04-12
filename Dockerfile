FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and configuration files
# COPY src/ ./src/
COPY config/ ./config/
COPY entrypoint.sh ./entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x ./entrypoint.sh

# Set environment variables for configuration

ARG SYNC_INTERVAL
ARG LOG_LEVEL

ENV SYNC_INTERVAL=$SYNC_INTERVAL
ENV LOG_LEVEL=$LOG_LEVEL

# Set the entrypoint to the script
ENTRYPOINT ["./entrypoint.sh"]