# Emoji CalDAV Sync

I like adding emojis to my calendar events and tasks to make them easy to spot. But doing it manually takes time! This project syncs events from a CalDAV calendar and makes sure every event starts with an emoji, adding one if needed. It uses a predefined emoji dictionary that updates automatically by matching words to emojis. If you add an emoji to an event yourself, the dictionary learns and adds it for future use.

## Status

!!!This is a work in progress!!! 
- The python script runs, is not fully tested
- Not tested in a docker environmet
- It currently only syncs events from calendar, not tasks
- It can only do one calendar, not multiple

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
│   ├── config.json        # Configuration file for CalDAV credentials
│   ├── emoji_dict.json    # Emoji dictionary for mapping words to emojis
├── sync.py                # Main script for syncing events
├── Dockerfile             # Docker configuration
├── entrypoint.sh          # Script to run the Python script periodically
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Configuration

### `config/config.json`

This file contains the CalDAV server configuration. Example:

```json
{
    "caldav_url": "https://your-caldav-server.com",
    "username": "your-username",
    "password": "your-password"
}
```

### `config/emoji_dict.json`

This file maps words to emojis. Example:

```json
{
    "meeting": "📅",
    "birthday": "🎂",
    "holiday": "🏖️"
}
```


## Contributing

Feel free to submit issues or pull requests to improve the project.


