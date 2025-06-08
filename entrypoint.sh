#!/bin/sh

# Default sleep interval to 3600 seconds (1 hour) if not set
SYNC_INTERVAL=${SYNC_INTERVAL:-3600}

# Run the Python script at specified intervals
while true; do
    python3 sync.py
    echo "Sleeping for $SYNC_INTERVAL seconds..."
    sleep "$SYNC_INTERVAL"  # Wait for the specified interval before running again
done