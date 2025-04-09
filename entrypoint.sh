#!/bin/sh

# Default sleep interval to 3600 seconds (1 hour) if not set
SYNC_INTERVAL=${SYNC_INTERVAL:-3600}

# Run the Python script at specified intervals
while true; do
    python3 sync.py --caldav_url "$CALDAV_URL" --username "$USERNAME" --password "$PASSWORD" --calendars_to_sync "$CALENDARS_TO_SYNC"
    sleep "$SYNC_INTERVAL"  # Wait for the specified interval before running again
done