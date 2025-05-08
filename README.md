# 🌟 Emoji CalDAV Sync 🌟

I love adding emojis to my calendar events and tasks to make them stand out! But doing it manually takes time. 🚀 This project syncs events from a CalDAV calendar and ensures every event starts with an emoji, adding one if needed. It uses a predefined emoji dictionary that updates automatically by matching words to emojis. If you add an emoji to an event yourself, the dictionary learns and adds it for future use. 🤖

## 🚧 Status
⚠️ This project is functional but has not been extensively tested. Use it with caution and report any issues you encounter. ⚠️

## ✨ Features

- 🔗 Connects to a CalDAV server using credentials from `config.json`.
- 🎨 Ensures events start with an emoji.
- ✅ Ensures tasks start with an emoji.
- 🔄 Automatically updates the emoji dictionary (`emoji_dict.json`) with new words and emojis.
- 🕒 Runs periodically using Docker.

## 📋 Prerequisites

- 🌐 A CalDAV server with valid credentials.
- 🐳 Docker (for running the script periodically).

## 🗂️ Project Structure

```
emoji-calldav/
├── config/
│   ├── emoji_dict.json    # 📖 Emoji dictionary for mapping words to emojis
│   ├── config.json        # 🔑 CalDAV configuration
├── sync.py                # 🛠️ Main script for syncing events
├── Dockerfile             # 🐳 Docker configuration
├── entrypoint.sh          # 🚀 Script to run the Python script periodically
├── requirements.txt       # 📦 Python dependencies
└── README.md              # 📚 Project documentation
```

## ⚙️ Configuration

### `config/config.json`

This file contains the configuration for syncing, including account details and calendars to sync. Example:

```json
{
    "CALDAV_URL": "https://example.com/remote.php/dav/calendars/username/",
    "USERNAME": "your_username@example.com",
    "PASSWORD": "your_secure_password",
    "CALENDARS_TO_SYNC": ["calendar1", "calendar2"]
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

- If this file doesn't exist, the code will create an empty dictionary for you. 🆕  
- The emojis from the CalDAV source take precedence. 🥇  
- If an existing word with a different emoji is loaded from the source, the dictionary is updated. 🔄  
- This code only changes an existing emoji in the CalDAV source if it is a '❓' emoji. ❓➡️🎉

### 🐳 Docker Arguments for Configuration

Example Docker run command:

```bash
docker run withoutanickname/caldav-emojifier
```

Or with a custom `SYNC_INTERVAL` and/or `LOG_LEVEL`:

```bash
docker run -e SYNC_INTERVAL=60 -e LOG_LEVEL=DEBUG caldavemojifier
```

### 🌍 Environment Variables

- `SYNC_INTERVAL`: ⏱️ Optional. The interval in seconds between sync operations. Defaults to `3600` seconds (1 hour) if not specified.
- `LOG_LEVEL`: 🛠️ Optional. Sets the logging level (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`).
- `MASTER_ON_COLLISION`: ⚖️ Optional. Determines which source takes precedence when there is a conflict between the emoji in the CalDAV event and the emoji in the emoji dictionary.  
    - If set to `CALDAV`, the emoji from the CalDAV event will overwrite the dictionary.  
    - If set to `EMOJI_DICT`, the emoji from the dictionary will overwrite the CalDAV event.  
    - Defaults to `CALDAV` if not specified.

This setup allows you to sync multiple calendars by specifying their names in the `CALDAV_CALENDARS` variable. 📅

## 🤝 Contributing

Feel free to submit issues or pull requests to improve the project. 💡
