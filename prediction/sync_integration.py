"""
Integration example for sync.py
===============================
Shows how to integrate the enhanced emoji predictor into your CalDAV sync process.
"""

# Add these imports to your sync.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm'))

try:
    from emoji_predictor import EmojiPredictor, predict_emoji_for_text
    ENHANCED_PREDICTION_AVAILABLE = True
    print("✅ Enhanced emoji prediction loaded")
except ImportError as e:
    ENHANCED_PREDICTION_AVAILABLE = False
    print(f"⚠️ Enhanced prediction not available: {e}")

# Initialize the predictor once (put this after loading config)
if ENHANCED_PREDICTION_AVAILABLE:
    emoji_predictor = EmojiPredictor(config_path="./config")
    emoji_predictor.train()  # This will use cached models if available
    print("🤖 Emoji predictor trained and ready")


def enhanced_words_to_emoji(words):
    """
    Enhanced version of words_to_emoji that combines ML prediction with dictionary lookup.
    Drop-in replacement for your existing words_to_emoji function.
    """
    if not ENHANCED_PREDICTION_AVAILABLE:
        # Fall back to original function
        return words_to_emoji_original(words)
    
    # Convert words list to text string
    if isinstance(words, list):
        text = " ".join(str(word) for word in words if word)
    else:
        text = str(words)
    
    # Use enhanced predictor
    predicted_emoji = emoji_predictor.predict_emoji(text)
    
    # If we get a good prediction, update the dictionary and return it
    if predicted_emoji != "❓":
        # Add to dictionary for future use (similar to your existing logic)
        first_word = sanitize_word(words[0]) if isinstance(words, list) and words else sanitize_word(text.split()[0] if text.split() else "")
        if first_word and first_word not in emoji_dict:
            update_or_add_word_to_emoji_dict(first_word, predicted_emoji)
        return predicted_emoji
    
    # Fall back to original logic if no good prediction
    return words_to_emoji_original(words)


def enhanced_add_words_to_emoji_dict(words, emoji="❓"):
    """
    Enhanced version that uses ML to suggest better emojis than the default ❓.
    """
    if not ENHANCED_PREDICTION_AVAILABLE:
        return add_words_to_emoji_dict_original(words, emoji)
    
    # If we're being asked to add with default emoji, try to predict a better one
    if emoji == "❓":
        text = " ".join(str(word) for word in words if word) if isinstance(words, list) else str(words)
        predicted_emoji = emoji_predictor.predict_emoji(text)
        if predicted_emoji != "❓":
            emoji = predicted_emoji
            logger.info(f"Enhanced prediction suggested {emoji} for '{text}'")
    
    # Use original logic with potentially enhanced emoji
    return add_words_to_emoji_dict_original(words, emoji)


# Backup original functions before replacing
words_to_emoji_original = words_to_emoji
add_words_to_emoji_dict_original = add_words_to_emoji_dict

# Replace with enhanced versions
words_to_emoji = enhanced_words_to_emoji
add_words_to_emoji_dict = enhanced_add_words_to_emoji_dict


# Usage in your existing process_event and process_task functions:
# No changes needed! Just import this file and the functions will be enhanced automatically.

print("🔄 Enhanced emoji prediction integrated successfully")