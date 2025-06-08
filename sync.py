import os
from dotenv import load_dotenv
import sys
import json
import logging
from datetime import date
from datetime import datetime
from datetime import timedelta

sys.path.insert(0, "..")
sys.path.insert(0, ".")

import caldav

# Load environment variables from .devcontainer.env
load_dotenv(dotenv_path=".devcontainer/.devcontainer.env")
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
master_on_collision = os.getenv('MASTER_ON_COLLISION', 'CALDAV').upper()
sync_interval = os.getenv('SYNC_INTERVAL', '3600')  

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.info("LOG_LEVEL: %s, SYNC_INTERVAL: %s, MASTER_ON_COLLISION: %s", log_level, sync_interval, master_on_collision)

# get arguments
# Load configuration from config.json
config_path = "./config/config.json"
if not os.path.exists(config_path):
    logger.error("Configuration file not found at '%s'. Please create a config.json file.", config_path)
    sys.exit(1)

with open(config_path, "r", encoding="utf-8") as config_file:
    config = json.load(config_file)

caldav_url = config.get("CALDAV_URL")
username = config.get("USERNAME")
password = config.get("PASSWORD")
calendars_to_sync = config.get("CALENDARS_TO_SYNC", [])
if not isinstance(calendars_to_sync, list):
    logger.error("Expected 'CALENDARS_TO_SYNC' to be a list in config.json.")
    sys.exit(1)

if not caldav_url or not username or not password:   
    logger.error("Missing required configuration values in config.json.")
    sys.exit(1)
headers = {"X-MY-CUSTOMER-HEADER": "123"}




# Check if emoji_dict.json exists, create it if it doesn't
emoji_dict_path = "config/emoji_dict.json"
if not os.path.exists(emoji_dict_path):
    logger.info("Emoji dictionary not found. Creating an empty emoji_dict.json.")
    os.makedirs(os.path.dirname(emoji_dict_path), exist_ok=True)
    with open(emoji_dict_path, "w", encoding="utf-8") as emoji_file:
        json.dump({}, emoji_file, indent=4, ensure_ascii=False)

# Load emoji dictionary
with open(emoji_dict_path, "r", encoding="utf-8") as emoji_file:
    emoji_dict = json.load(emoji_file)

def sanitize_word(word):
    word = ''.join(char for char in word if char.isalpha())
    #word = ''.join(char for char in word if char.isalnum() or char.isspace() or char in "!@#$%^&*()-_=+[]{}|;:'\",.<>?/\\")
    return word.upper()

def is_emoji(character):
    """
    Somewhat check if this is an emoji
    """    
    character = character.encode("unicode-escape").decode("utf-8") #decode it to utf-8
    return (character.startswith("\\U") or character.startswith("\\u")) # if the character starts with \U or \u, it is an emoji
    
def print_calendars(calendars):
    if calendars:
        logger.info("Your principal has %i calendars:", len(calendars))
        for c in calendars:
            logger.debug("    Name: %-36s  URL: %s", c.name, c.url)
    else:
        logger.info("Your principal has no calendars")

def update_or_add_word_to_emoji_dict(word,emoji):
    """
    Update the emoji_dict with a new word and emoji.
    If the word already exists in the emoji_dict, no changes are made.
    """
    if word in emoji_dict:
        logger.info("Updating '%s' emoji from '%s' to '%s'.", word, emoji_dict[word], emoji)
    else:
        logger.info("Adding new word '%s' with emoji '%s'.", word, emoji)
    emoji_dict[word] = emoji
    with open("config/emoji_dict.json", "w", encoding="utf-8") as emoji_file:
        json.dump(emoji_dict, emoji_file, indent=4, ensure_ascii=False)
    return True

def add_words_to_emoji_dict(words, emoji="❓"):
    """
    Add a word to the emoji_dict with the given emoji.
    If the word already exists in the emoji_dict, no changes are made.
    It returns True if a word was found, False if it is a new word.
    """
    first_word=sanitize_word(words[0])
    found = False
    for word in words:
        word = sanitize_word(word)
        if not word:
            logger.debug("Skipping empty word")
            continue #go to the next word
        elif first_word=="":
                logger.debug("first word is empty, setting it to '%s'", word)
                first_word=word #the first word is the one we want to add
        
        if word in emoji_dict:   
            found=True         
            if (emoji_dict[word] != emoji):
                # word does exist in emoji_dict but with a different emoji
                if (emoji == "❓"):
                    # We don't change back to the default emoji
                    logger.debug("Trying to update '%s' emoji to default. Current emoji is '%s'. doing nothing", word, emoji_dict[word])
                elif ((master_on_collision == "EMOJI_DICT") and  (emoji_dict[word] != "❓")):
                    # The emoji_dict is the master, so we don't change it
                    logger.debug("'%s' already exists in emoji_dict with emoji '%s', emoji_dict is master, do nothing.'%s'.", word, emoji_dict[word], emoji)
                else:
                    # The emoji is set in the emoji_dict, but now we have a new one. Update it to the new emoji
                    logger.debug("'%s' already exists in emoji_dict with emoji '%s', updating to '%s'.", word, emoji_dict[word], emoji)
                    update_or_add_word_to_emoji_dict(word, emoji)
            else:
                logger.debug("'%s' already exists in emoji_dict with the same emoji '%s'.", word, emoji)
                # No changes needed, just log it
            
            return found #if one word is found, we can stop checking the rest of the words
    #end of for word in words loop

    # If none of the words are found, add the first word
    if not found and first_word:
        logger.debug("None of the words exist in emoji_dict. Adding first word '%s' with emoji '%s'.", first_word, emoji)
        update_or_add_word_to_emoji_dict(first_word, emoji)
            


def words_to_emoji(words):
    """
    Convert a word to an emoji using the emoji_dict.
    If the word is not in the emoji_dict, add it with a default emoji.
    this function will always return a emoji, even if it is the default one.
    """
    found = False
    for word in words:
        word = sanitize_word(word)
        # check each word
        if word in emoji_dict:
            logger.debug("Found '%s' in emoji_dict with emoji '%s'.", word, emoji_dict[word])
            return emoji_dict[word]
            

    # if no word is found, add the first word to the emoji_dict with a default emoji
    if not found:
        logger.debug("Adding '%s' to emoji_dict with default emoji.", words)
        add_words_to_emoji_dict(words)

    return "❓"

def process_event(event):
    """Process a single calendar event."""
    
    event_name = event.icalendar_component.get("summary")
    logger.debug("##Event: %s", event_name)
    #if event_name[0].isalpha(): #the event name starts with a letter
    if not is_emoji(event_name[0]):
        logger.debug("This event does not start with an emoji, let's add that")
        summary_parts = event_name.split(" ")
        logger.debug("Words: %s", summary_parts)
        emoji = words_to_emoji(summary_parts)
        logger.info("Event %s, updated emoji to: %s",event.icalendar_component.get("summary"), emoji)
        event.vobject_instance.vevent.summary.value = emoji + " " + event_name
        event.save()
    elif (event_name[0] == "❓"):
        logger.debug("Event '%s' starts with the default emoji '%s'.", event_name, event_name[0])
        # The event name starts with the default emoji, we can still try and add a emoji to this word
        event_name = event_name[1:]  # Remove the first character (default emoji) from the event name
        summary_parts = event_name.split(" ")
        emoji = words_to_emoji(summary_parts)
        if not emoji:
            # no emoji found, do nothing (already has the default emoji)
            pass
        elif emoji == "❓":
            # no emoji found, do nothing (already has the default emoji)
            pass
        elif emoji:
            logger.info("Event %s, updated default emoji to: %s",event.icalendar_component.get("summary"), emoji)
            event.vobject_instance.vevent.summary.value = emoji + " " + event_name.lstrip()
            event.save()
    else:
        logger.debug("This event already starts with an emoji, check if the word and emoji is known")
        if(event_name[1] == " "):
            # If the event name starts with an emoji followed by a space, split the string on spaces
            summary_parts = event_name.split(" ")            
            emoji = summary_parts.pop(0) # pop the first character which is the emoticon
            words = summary_parts #the rest of the string are the words
            logger.debug("splitting with spaces. Emoji: %s Words: %s", emoji, words)
        else:
            # If the event name starts with an emoji followed by a word, split the string
            summary_parts = event_name.split(" ")
            emoji = summary_parts[0][0]  # Get the first character (emoji) from the first item
            summary_parts[0] = summary_parts[0][1:]  # Remove the first character (emoji) from the string
            words = summary_parts 
            logger.debug("splitting no space. Emoji: %s Words: %s", emoji, words)
        event_name = event_name[1:]  # Remove the first character (emoji) from the event name
        current_emoji = emoji
        emoji_from_dict = words_to_emoji(words)
        if master_on_collision == "EMOJI_DICT" and emoji_from_dict != "❓":
            # the emoji does not match the emoji_dict
            # since emoji_dict is the master, we update the caldav event with the emoji_dict emoji
            if current_emoji != emoji_from_dict:
                logger.debug("'%s' already exists in emoji_dict with emoji '%s', emoji_dict is master, update caldav.'%s'.", words, current_emoji, emoji_from_dict)
                logger.info("Event %s, updated emoji to: %s because emoji_dict is master",event.icalendar_component.get("summary"), emoji_from_dict)
                event.vobject_instance.vevent.summary.value = emoji_from_dict + " " + event_name.lstrip()
                event.save()
        else:
            add_words_to_emoji_dict(words, emoji)

def process_task(task):
    """Process a single task."""
    task_name = task.icalendar_component.get("summary")
    logger.debug("##Task: %s", task_name)
    if not is_emoji(task_name[0]):
        logger.debug("This task does not start with an emoji, let's add that")
        emoji = words_to_emoji(task_name.split(" "))
        logger.info("Task %s, updated emoji to: %s", task_name, emoji)
        task.vobject_instance.vtodo.summary.value = emoji + " " + task_name    
        task.save()
    elif task_name[0] == "❓":
        # The task name starts with the default emoji, we can still try and add an emoji to this word
        task_name = task_name[1:]  # Remove the first character (default emoji) from the task name
        summary_parts = task_name.split(" ")
        emoji = words_to_emoji(summary_parts)
        if emoji and emoji != "❓":
            logger.info("Task %s, updated emoji to: %s", task_name, emoji)
            task.vobject_instance.vtodo.summary.value = emoji + " " + task_name.lstrip()
            task.save()
    else:
        logger.debug("This task already starts with an emoji, check if the word and emoji is known and/or add it to the emoji_dict")
        if task_name[1] == " ":
            # If the task name starts with an emoji followed by a space, split the string on spaces
            summary_parts = task_name.split(" ")
            emoji = summary_parts.pop(0)  # Pop the first character which is the emoji
            words = summary_parts  # The rest of the string are the words
            logger.debug("Splitting with spaces. Emoji: %s Words: %s", emoji, words)
        else:
            # If the task name starts with an emoji followed by a word, split the string
            summary_parts = task_name.split(" ")
            emoji = summary_parts[0][0]  # Get the first character (emoji) from the first item
            summary_parts[0] = summary_parts[0][1:]  # Remove the first character (emoji) from the string
            words = summary_parts
            logger.debug("Splitting no space. Emoji: %s Words: %s", emoji, words)
        task_name = task_name[1:]  # Remove the first character (emoji) from the event name
        current_emoji = emoji
        emoji_from_dict = words_to_emoji(words)
        if master_on_collision == "EMOJI_DICT" and emoji_from_dict != "❓":
            # the emoji does not match the emoji_dict
            # since emoji_dict is the master, we update the caldav event with the emoji_dict emoji
            if current_emoji != emoji_from_dict:
                logger.debug("'%s' already exists in emoji_dict with emoji '%s', emoji_dict is master, update caldav.'%s'.", words, current_emoji, emoji_from_dict)
                logger.info("Task %s, updated emoji to: %s because emoji_dict is master",task_name, emoji_from_dict)
                task.vobject_instance.vtodo.summary.value = emoji_from_dict + " " + task_name.lstrip()
                task.save()
        else:
            add_words_to_emoji_dict(words, emoji)


if __name__ == "__main__":
    with caldav.DAVClient(
        url=caldav_url,
        username=username,
        password=password,
        headers=headers,
    ) as client:
        logger.info("connecting")
        my_principal = client.principal()
        calendars = my_principal.calendars()
        for calendar_to_sync in calendars_to_sync:
            try:
                current_calendar = my_principal.calendar(name=calendar_to_sync)
            except caldav.error.NotFoundError:
                logger.error("Calendar with name '%s' not found. Skipping.", calendar_to_sync)
                logger.info("Available calendars:")
                for c in calendars:
                    logger.info("    Name: %-36s ", getattr(c, 'name', 'Unknown'))
                continue
            except Exception as e:
                logger.error("Error accessing calendar '%s': %s. Skipping.", calendar_to_sync, str(e))
                continue

            events = current_calendar.search(
                start=datetime.now(),
                end=datetime(date.today().year + 1, 1, 1),
                event=True,
                expand=False,
            )
            tasks = current_calendar.search(
                start=datetime.now(),
                end=datetime(date.today().year + 1, 1, 1),
                todo=True,
                expand=False,
            )
            logger.info("Found %i events and %i tasks in calendar '%s'", len(events), len(tasks), calendar_to_sync)
            for event in events:
                process_event(event)
                logger.debug("/////////next event/////////")
            for task in tasks:
                process_task(task)
                logger.debug("/////////next task/////////")

        logger.info("sync done")