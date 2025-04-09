FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and configuration files
COPY src/ ./src/
COPY config/ ./config/
COPY entrypoint.sh .

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Set environment variables for configuration
ARG CALDAV_URL
ARG USERNAME
ARG PASSWORD
ARG CALENDARS_TO_SYNC
ARG SYNC_INTERVAL

ENV CALDAV_URL=$CALDAV_URL
ENV USERNAME=$USERNAME
ENV PASSWORD=$PASSWORD
ENV CALENDARS_TO_SYNC=$CALENDARS_TO_SYNC
ENV SYNC_INTERVAL=$SYNC_INTERVAL

# Set the entrypoint to the script
ENTRYPOINT ["./entrypoint.sh"]