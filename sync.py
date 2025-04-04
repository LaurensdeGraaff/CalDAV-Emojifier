import sys
import json
import logging
from datetime import date
from datetime import datetime
from datetime import timedelta

sys.path.insert(0, "..")
sys.path.insert(0, ".")

import caldav

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

with open("config/config.json", "r") as config_file:
    config = json.load(config_file)
with open("config/emoji_dict.json", "r") as emoji_file:
    emoji_dict = json.load(emoji_file)

caldav_url = config["caldav_url"]
username = config["username"]
password = config["password"]
headers = {"X-MY-CUSTOMER-HEADER": "123"}
calendar_to_sync = config["calendar_to_sync"]

def print_calendars(calendars):
    if calendars:
        logger.info("Your principal has %i calendars:", len(calendars))
        for c in calendars:
            logger.info("    Name: %-36s  URL: %s", c.name, c.url)
    else:
        logger.info("Your principal has no calendars")


def add_word_to_emoji_dict(word, emoji="❓"):
    if word in emoji_dict:
        logger.info("'%s' already exists in emoji_dict. No changes made.", word)
        return

    emoji_dict[word] = emoji
    logger.info("Adding '%s' to emoji_dict with emoji '%s'.", word, emoji)

    with open("config/emoji_dict.json", "w", encoding="utf-8") as emoji_file:
        json.dump(emoji_dict, emoji_file, indent=4, ensure_ascii=False)


def word_to_emoji(word):
    if word not in emoji_dict:
        add_word_to_emoji_dict(word)

    return emoji_dict[word]


if __name__ == "__main__":
    with caldav.DAVClient(
        url=caldav_url,
        username=username,
        password=password,
        headers=headers,
    ) as client:
        my_principal = client.principal()
        calendars = my_principal.calendars()
        print_calendars(calendars)

        dev_calendar = my_principal.calendar(name=calendar_to_sync)
        logger.info("Working with calendar: %s", dev_calendar.name)
        events = dev_calendar.search(
            start=datetime.now(),
            end=datetime(date.today().year + 1, 1, 1),
            event=True,
            expand=False,
        )
        logger.info("Found %i events", len(events))
        for event in events:
            logger.info("##Event: %s", event.icalendar_component.get("summary"))
            if event.icalendar_component.get("summary")[0].isalpha():
                logger.info("This event does not start with an emoji, let's add that")

                assert len(event.wire_data) >= len(event.data)
                emoji = word_to_emoji(event.icalendar_component.get("summary").split()[0])

                event.vobject_instance.vevent.summary.value = emoji + " " + event.icalendar_component.get("summary")

                event.save()
            else:
                # Check if the event summary has at least 2 words
                summary_parts = event.icalendar_component.get("summary").split()
                if len(summary_parts) >= 2:
                    emoji = summary_parts[0]
                    word = summary_parts[1]
                else:
                    # If not, treat the first character as the emoji and the rest as the word
                    emoji = event.icalendar_component.get("summary")[0]
                    word = event.icalendar_component.get("summary")[1:]

                logger.info("This event already starts with an emoji, check if the word and emoji is known and/or add it to the emoji_dict")
                logger.info("Emoji: %s", emoji)
                logger.info("Word: %s", word)
                add_word_to_emoji_dict(word, emoji)
            logger.info("")

        logger.info("End of script")