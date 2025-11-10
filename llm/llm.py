"""
Emoji Classifier
================
Train a Logistic Regression classifier to predict emojis based on text phrases.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import json
import os
import logging
import re


# Configure logging
logging.basicConfig(level=getattr(logging, 'INFO', logging.DEBUG), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def is_emoji(character):
    """
    Somewhat check if this is an emoji
    """    
    character = character.encode("unicode-escape").decode("utf-8") #decode it to utf-8
    return (character.startswith("\\U") or character.startswith("\\u")) # if the character starts with \U or \u, it is an emoji
  

def parse_calendar_data():
    """Parse the training.ics file and extract event summaries that start with emojis."""
    
    # Read the training.ics file
    with open("training.ics", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract all SUMMARY lines using regex
    summary_pattern = r'SUMMARY:(.+)'
    summaries = re.findall(summary_pattern, content)
    
    # Filter summaries that start with emoji and split them
    emoji_summaries = []
    seen_combinations = set()  # Track (summary, emoji) pairs
    
    for summary in summaries:
        # Remove any leading/trailing whitespace
        cleaned = summary.strip()
        if cleaned and is_emoji(cleaned[0]):
            # Split emoji from the rest of the text
            emoji = cleaned[0]
            # Skip if emoji is '❓'
            if emoji == "❓":
                continue
            text_without_emoji = cleaned[1:].strip()
            text_without_emoji = sanitize_text(text_without_emoji)
            if text_without_emoji:  # Only add if there's text after the emoji
                # Create a unique key for this combination
                combination_key = (text_without_emoji, emoji)
                if combination_key not in seen_combinations:
                    seen_combinations.add(combination_key)
                    emoji_summaries.append({"summary": text_without_emoji, "emoji": emoji})
    
    return emoji_summaries

def sanitize_text(text):
    sanitized = ''.join(char for char in text if char.isalpha() or char.isspace())
    return sanitized.upper()

def create_calendar_training_data():
    """Create training data from calendar events."""
    summaries = parse_calendar_data()
    
    print(f"Found {len(summaries)} calendar events:")
    for i, summary in enumerate(summaries[:50], 1):  # Show first 50
        print(f"  {i}. {summary}")
    if len(summaries) > 50:
        print(f"  ... and {len(summaries) - 50} more")
    
    return summaries

def create_training_data():
    """Create a small training dataset with phrases and corresponding emojis."""
    # Training data: (phrase, emoji)

    data = create_calendar_training_data()
    X = [item["summary"] for item in data]
    y = [item["emoji"] for item in data]
    
    return X, y


def train_classifier(X_train, y_train):
    """
    Train a Logistic Regression classifier with TF-IDF vectorization.
    
    Args:
        X_train: List of text phrases
        y_train: List of corresponding emojis
        
    Returns:
        Trained pipeline (vectorizer + classifier)
    """
    # Create pipeline with TF-IDF vectorizer and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            max_features=100
        )),
        ('classifier', LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    return pipeline


def predict_emoji(classifier, text):
    """
    Predict emoji for a given text.
    
    Args:
        classifier: Trained classifier pipeline
        text: Input text phrase
        
    Returns:
        Predicted emoji
    """
    prediction = classifier.predict([text])[0]
    return prediction


def predict_with_probabilities(classifier, text):
    """
    Predict emoji with probability scores.
    
    Args:
        classifier: Trained classifier pipeline
        text: Input text phrase
        
    Returns:
        Tuple of (predicted emoji, probability dict)
    """
    prediction = classifier.predict([text])[0]
    probabilities = classifier.predict_proba([text])[0]
    classes = classifier.classes_
    
    prob_dict = {emoji: prob for emoji, prob in zip(classes, probabilities)}
    
    return prediction, prob_dict


def main():
    """Main function to demonstrate the emoji classifier."""
    print("🎯 Emoji Classifier Demo (Logistic Regression)\n")
    print("=" * 50)
    
    # Create training data
    print("\n📊 Creating training data...")
    X_train, y_train = create_training_data()
    print(f"   Training samples: {len(X_train)}")
    print(f"   Unique emojis: {len(set(y_train))}")
    print(f"   Emojis: {', '.join(set(y_train))}")
    
    # Train classifier
    print("\n🔧 Training Logistic Regression classifier...")
    classifier = train_classifier(X_train, y_train)
    print("   Training complete!")
    
    # Test predictions
    print("\n" + "=" * 50)
    print("🧪 Testing predictions:\n")
    
    test_phrases = [
        "whisky drinken",
        "huisarts",
        "tandarts",
        "werken in Arnhem",
        "biertje drinken in Arnhem",
        "afspraak bij de dokter",
        "vergadering op maandag",
        "vakantie vieren",
        "kiespijn naar tandarts",
        "productie deadline",
        "frikandelbroodje eten"
    ]
    
    for phrase in test_phrases:
        emoji, probs = predict_with_probabilities(classifier, phrase)
        
        # Sort probabilities by value
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        print(f"📝 '{phrase}'")
        print(f"   → Predicted: {emoji} (confidence: {probs[emoji]:.2%})")
        
        # Show top 2 probabilities
        if len(sorted_probs) > 1:
            print(f"   Probabilities: ", end="")
            print(" | ".join([f"{e}: {p:.1%}" for e, p in sorted_probs]))
        print()
    
    # Interactive mode
    print("=" * 50)
    print("\n💬 Interactive mode (type 'quit' to exit):\n")
    
    while True:
        user_input = input("Enter a phrase: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        emoji = predict_emoji(classifier, user_input)
        print(f"   → {emoji}\n")

if __name__ == "__main__":
    main()