"""
Enhanced Emoji Classifier
=========================
Train multiple classifiers and combine with existing emoji dictionary for best predictions.
Uses both training.ics data and emoji_dict.json for optimal performance.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import json
import os
import logging
import re
from collections import Counter, defaultdict


# Configure logging
logging.basicConfig(level=getattr(logging, 'INFO', logging.DEBUG), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def is_emoji(character):
    """
    Check if this is an emoji
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
    """Clean and normalize text for consistent processing."""
    sanitized = ''.join(char for char in text if char.isalpha() or char.isspace())
    return sanitized.upper()


def load_emoji_dict():
    """Load the existing emoji dictionary."""
    emoji_dict_path = "../config/emoji_dict.json"
    if os.path.exists(emoji_dict_path):
        with open(emoji_dict_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def create_calendar_training_data():
    """Create training data from calendar events."""
    summaries = parse_calendar_data()
    
    print(f"Found {len(summaries)} calendar events:")
    for i, summary in enumerate(summaries[:10], 1):  # Show first 10
        print(f"  {i}. {summary}")
    if len(summaries) > 10:
        print(f"  ... and {len(summaries) - 10} more")
    
    return summaries


def create_dictionary_training_data():
    """Create training data from the emoji dictionary."""
    emoji_dict = load_emoji_dict()
    training_data = []
    
    for word, emoji in emoji_dict.items():
        if emoji != "❓":  # Skip default emoji entries
            training_data.append({"summary": word, "emoji": emoji})
    
    print(f"\nFound {len(training_data)} entries in emoji dictionary (excluding ❓)")
    return training_data


def augment_training_data(word_emoji_pairs):
    """Augment training data by creating variations and combinations."""
    augmented = []
    
    for item in word_emoji_pairs:
        text = item["summary"]
        emoji = item["emoji"]
        
        # Add original
        augmented.append(item)
        
        # Add variations with common prefixes/suffixes
        variations = [
            text,
            f"NAAR {text}",
            f"{text} DOEN", 
            f"{text} AVOND",
            f"AFSPRAAK {text}",
            f"{text} BEZOEKEN"
        ]
        
        for variation in variations:
            if variation != text:  # Don't duplicate original
                augmented.append({"summary": variation, "emoji": emoji})
    
    return augmented

def create_combined_training_data():
    """Combine both calendar data and emoji dictionary for comprehensive training."""
    # Get data from both sources
    calendar_data = create_calendar_training_data()
    dict_data = create_dictionary_training_data()
    
    # Combine the datasets
    all_data = calendar_data + dict_data
    
    # Augment with variations
    augmented_data = augment_training_data(all_data)
    
    # Deduplicate while preserving order
    seen = set()
    deduplicated = []
    for item in augmented_data:
        key = (item["summary"], item["emoji"])
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)
    
    X = [item["summary"] for item in deduplicated]
    y = [item["emoji"] for item in deduplicated]
    
    print(f"\nCombined training data: {len(X)} samples")
    print(f"Unique emojis: {len(set(y))}")
    
    # Show emoji distribution
    emoji_counts = Counter(y)
    print("\nTop 10 emoji frequencies:")
    for emoji, count in emoji_counts.most_common(10):
        print(f"  {emoji}: {count}")
    
    return X, y


def train_ensemble_classifier(X_train, y_train):
    """
    Train an ensemble of classifiers for better predictions.
    
    Args:
        X_train: List of text phrases
        y_train: List of corresponding emojis
        
    Returns:
        Dictionary of trained models
    """
    print("\n🔧 Training multiple classifiers...")
    
    # Create different feature extraction approaches
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
        max_features=500,
        min_df=1,
        max_df=0.95
    )
    
    count_vectorizer = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=300,
        min_df=1
    )
    
    # Individual classifiers
    models = {}
    
    # Logistic Regression with TF-IDF
    lr_tfidf = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    
    # Naive Bayes with Count Vectorizer  
    nb_count = Pipeline([
        ('count', count_vectorizer),
        ('classifier', MultinomialNB(alpha=1.0))
    ])
    
    # Random Forest with TF-IDF
    rf_tfidf = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=200)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
    ])
    
    # Train individual models
    models['logistic_tfidf'] = lr_tfidf
    models['naive_bayes'] = nb_count  
    models['random_forest'] = rf_tfidf
    
    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)
        
        # Quick cross-validation score
        if len(set(y_train)) > 1:  # Only if we have multiple classes
            scores = cross_val_score(model, X_train, y_train, cv=min(3, len(X_train)//2))
            print(f"   {name} CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Create ensemble
    ensemble = VotingClassifier([
        ('lr', lr_tfidf),
        ('nb', nb_count),
        ('rf', rf_tfidf)
    ], voting='soft')
    
    print("   Training ensemble...")
    ensemble.fit(X_train, y_train)
    models['ensemble'] = ensemble
    
    return models


def train_classifier(X_train, y_train):
    """Wrapper to maintain compatibility."""
    models = train_ensemble_classifier(X_train, y_train)
    return models['ensemble']  # Return the best performing model


class EmojiPredictor:
    """Enhanced emoji predictor that combines ML models with dictionary lookup."""
    
    def __init__(self):
        self.models = None
        self.emoji_dict = load_emoji_dict()
        self.word_to_emoji = self._create_word_mapping()
    
    def _create_word_mapping(self):
        """Create direct word-to-emoji mapping for fast lookup."""
        return {word.lower(): emoji for word, emoji in self.emoji_dict.items() if emoji != "❓"}
    
    def train(self, X_train, y_train):
        """Train the ensemble of models."""
        self.models = train_ensemble_classifier(X_train, y_train)
    
    def predict_emoji(self, text):
        """
        Predict emoji using multiple strategies and return the best result.
        
        Strategy:
        1. Direct word lookup in emoji dictionary
        2. ML model prediction
        3. Combination of both with confidence scoring
        """
        text_clean = sanitize_text(text)
        words = text_clean.split()
        
        # Strategy 1: Direct dictionary lookup
        dict_matches = []
        for word in words:
            if word.lower() in self.word_to_emoji:
                dict_matches.append((word, self.word_to_emoji[word.lower()]))
        
        # Strategy 2: ML prediction (if models are trained)
        ml_prediction = None
        ml_confidence = 0
        
        if self.models and len(text_clean.strip()) > 0:
            try:
                ensemble = self.models['ensemble']
                ml_prediction = ensemble.predict([text_clean])[0]
                
                # Get confidence from individual models
                confidences = []
                for name, model in self.models.items():
                    if name != 'ensemble':
                        try:
                            proba = model.predict_proba([text_clean])[0]
                            pred_idx = list(model.classes_).index(ml_prediction)
                            confidences.append(proba[pred_idx])
                        except:
                            pass
                
                ml_confidence = np.mean(confidences) if confidences else 0
            except Exception as e:
                logger.debug(f"ML prediction failed: {e}")
        
        # Strategy 3: Combine results
        if dict_matches and ml_prediction:
            # If we have both, prefer dictionary match for high confidence
            dict_emojis = [emoji for _, emoji in dict_matches]
            if ml_prediction in dict_emojis:
                return ml_prediction  # ML confirms dictionary
            elif ml_confidence > 0.7:
                return ml_prediction  # High ML confidence
            else:
                return dict_matches[0][1]  # Fall back to dictionary
        
        elif dict_matches:
            return dict_matches[0][1]  # Only dictionary match
        
        elif ml_prediction:
            return ml_prediction  # Only ML prediction
        
        else:
            return "❓"  # Default fallback
    
    def predict_with_probabilities(self, text):
        """Get prediction with detailed probability information."""
        text_clean = sanitize_text(text)
        
        result = {
            'prediction': self.predict_emoji(text),
            'dictionary_matches': [],
            'ml_predictions': {},
            'confidence': 0
        }
        
        # Check dictionary matches
        words = text_clean.split()
        for word in words:
            if word.lower() in self.word_to_emoji:
                result['dictionary_matches'].append({
                    'word': word,
                    'emoji': self.word_to_emoji[word.lower()]
                })
        
        # Get ML predictions from all models
        if self.models and len(text_clean.strip()) > 0:
            for name, model in self.models.items():
                try:
                    prediction = model.predict([text_clean])[0]
                    probabilities = model.predict_proba([text_clean])[0]
                    classes = model.classes_
                    
                    prob_dict = {emoji: prob for emoji, prob in zip(classes, probabilities)}
                    max_prob = max(prob_dict.values())
                    
                    result['ml_predictions'][name] = {
                        'prediction': prediction,
                        'confidence': max_prob,
                        'top_3': sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                    }
                except Exception as e:
                    logger.debug(f"Failed to get prediction from {name}: {e}")
        
        # Calculate overall confidence
        if result['dictionary_matches']:
            result['confidence'] += 0.5
        
        if result['ml_predictions']:
            avg_ml_confidence = np.mean([
                pred['confidence'] for pred in result['ml_predictions'].values()
            ])
            result['confidence'] += avg_ml_confidence * 0.5
        
        return result


# Backward compatibility functions
def predict_emoji(classifier, text):
    """Legacy function for backward compatibility."""
    if hasattr(classifier, 'predict_emoji'):
        return classifier.predict_emoji(text)
    else:
        return classifier.predict([text])[0]


def predict_with_probabilities(classifier, text):
    """Legacy function for backward compatibility.""" 
    if hasattr(classifier, 'predict_with_probabilities'):
        return classifier.predict_with_probabilities(text)
    else:
        prediction = classifier.predict([text])[0]
        probabilities = classifier.predict_proba([text])[0]
        classes = classifier.classes_
        prob_dict = {emoji: prob for emoji, prob in zip(classes, probabilities)}
        return prediction, prob_dict


def main():
    """Main function to demonstrate the enhanced emoji classifier."""
    print("🎯 Enhanced Emoji Classifier Demo\n")
    print("=" * 60)
    
    # Initialize predictor
    predictor = EmojiPredictor()
    
    # Create comprehensive training data
    print("\n📊 Creating combined training data...")
    X_train, y_train = create_combined_training_data()
    
    if len(X_train) > 0:
        # Train the predictor
        print(f"\n🔧 Training enhanced classifier with {len(X_train)} samples...")
        predictor.train(X_train, y_train)
        print("   Training complete!")
    else:
        print("\n⚠️  No training data available, using dictionary-only mode")
    
    # Test predictions
    print("\n" + "=" * 60)
    print("🧪 Testing predictions:\n")
    
    test_phrases = [
        "WHISKY DRINKEN",
        "HUISARTS BEZOEKEN", 
        "TANDARTS AFSPRAAK",
        "WERKEN IN ARNHEM",
        "BIERTJE DRINKEN",
        "KAPPER AFSPRAAK",
        "VERGADERING OP KANTOOR",
        "VAKANTIE PLANNEN",
        "SPELLETJESAVOND ORGANISEREN",
        "KERK DIENST",
        "BOODSCHAPPEN DOEN",
        "CONCERT BEZOEKEN",
        "FRIKANDELBROODJE ETEN"
    ]
    
    for phrase in test_phrases:
        result = predictor.predict_with_probabilities(phrase)
        
        print(f"📝 '{phrase}'")
        print(f"   → Predicted: {result['prediction']} (confidence: {result['confidence']:.2f})")
        
        # Show dictionary matches
        if result['dictionary_matches']:
            matches_str = ', '.join([f"{m['word']}→{m['emoji']}" for m in result['dictionary_matches']])
            print(f"   📚 Dictionary matches: {matches_str}")
        
        # Show top ML predictions
        if result['ml_predictions']:
            best_ml = max(result['ml_predictions'].items(), key=lambda x: x[1]['confidence'])
            print(f"   🤖 Best ML: {best_ml[1]['prediction']} ({best_ml[1]['confidence']:.2%} from {best_ml[0]})")
        
        print()
    
    # Show model performance summary
    if predictor.models:
        print("=" * 60)
        print("📈 Model Performance Summary:")
        print(f"   Dictionary entries: {len(predictor.emoji_dict)}")
        print(f"   Active mappings: {len(predictor.word_to_emoji)}")
        print(f"   Trained models: {len(predictor.models)}")
        
    # Interactive mode
    print("\n💬 Interactive mode (type 'quit' for exit, 'detail' for detailed info):\n")
    
    while True:
        user_input = input("Enter a phrase: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'detail':
            # Show detailed prediction for last input
            continue
        
        if not user_input:
            continue
        
        # Simple prediction
        emoji = predictor.predict_emoji(user_input)
        print(f"   → {emoji}")
        
        # Ask for detailed view
        detail = input("     Show details? (y/n): ").strip().lower()
        if detail == 'y':
            result = predictor.predict_with_probabilities(user_input)
            print(f"\n   🔍 Detailed Analysis:")
            print(f"   Final prediction: {result['prediction']}")
            print(f"   Overall confidence: {result['confidence']:.2f}")
            
            if result['dictionary_matches']:
                print(f"   Dictionary matches:")
                for match in result['dictionary_matches']:
                    print(f"     • {match['word']} → {match['emoji']}")
            
            if result['ml_predictions']:
                print(f"   ML model predictions:")
                for name, pred in result['ml_predictions'].items():
                    print(f"     • {name}: {pred['prediction']} ({pred['confidence']:.2%})")
                    if pred['top_3']:
                        top_3 = ", ".join([f"{e}:{p:.1%}" for e, p in pred['top_3'][:3]])
                        print(f"       Top 3: {top_3}")
        print()

if __name__ == "__main__":
    main()