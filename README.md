# Emoji CalDAV Sync

I like adding emojis to my calendar events and tasks to make them easy to spot. But doing it manually takes time! This project syncs events from a CalDAV calendar and makes sure every event starts with an emoji, adding one if needed. It uses a predefined emoji dictionary that updates automatically by matching words to emojis. If you add an emoji to an event yourself, the dictionary learns and adds it for future use.

## Status

!!!This is a work in progress!!! 
- The python script runs, is not fully tested
- Not tested in a docker environmet

## Features

- Connects to a CalDAV server using credentials from `config.json`.
- Ensures events start with an emoji.
- Ensures tasks start with an emoji (work in progress).
- Automatically updates the emoji dictionary (`emoji_dict.json`) with new words and emojis.
- Runs periodically using Docker.

## Prerequisites

- Python 3 and `pip3` (already included in the dev container).
- A CalDAV server with valid credentials.
- Docker (for running the script periodically).

## Project Structure

```
emoji-calldav/
├── config/
│   ├── emoji_dict.json    # Emoji dictionary for mapping words to emojis
├── sync.py                # Main script for syncing events
├── Dockerfile             # Docker configuration
├── entrypoint.sh          # Script to run the Python script periodically
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Configuration

### Docker Arguments for Configuration

Example Docker run command:

```bash
docker run -e CALDAV_URL="https://your-caldav-server.com" \
           -e CALDAV_USERNAME="your-username" \
           -e CALDAV_PASSWORD="your-password" \
           -e CALDAV_CALENDARS="calendar1,calendar2,calendar3" \
           emoji-caldav-sync
```

### Environment Variables

- `CALDAV_URL`: The URL of your CalDAV server.
- `CALDAV_USERNAME`: Your CalDAV username.
- `CALDAV_PASSWORD`: Your CalDAV password.
- `CALDAV_CALENDARS`: A comma-separated list of calendar names to sync.
- `LOG_LEVEL`: Optional. Sets the logging level (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`).

This setup allows you to sync multiple calendars by specifying their names in the `CALDAV_CALENDARS` variable.

### `config/emoji_dict.json`

This file maps words to emojis. Example:

```json
{
    "meeting": "📅",
    "birthday": "🎂",
    "holiday": "🏖️"
}
```

If this file doesn't exist, the code will create an empty dictionary for you.  
The emojis from the CalDAV source take precedence.  
If an existing word with a different emoji is loaded from the source, the dictionary is updated.  
This code only changes an existing emoji in the CalDAV source if it is a '❓' emoji.


## Contributing

Feel free to submit issues or pull requests to improve the project.
