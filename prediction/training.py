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
from sklearn.utils.class_weight import compute_class_weight
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


def create_fuzzy_matches(word):
    """Generate fuzzy variations of a word to handle typos and variations."""
    variations = set([word])
    
    # Common letter substitutions for typos
    substitutions = {
        'S': ['SK', 'Z', 'C'],
        'K': ['CK', 'C', 'Q'],
        'Y': ['IJ', 'I', 'EY'],
        'IJ': ['Y', 'I'],
        'C': ['K', 'CK', 'S'],
        'Z': ['S'],
        'PH': ['F'],
        'F': ['PH'],
        'E': ['EE', 'I'],
        'EE': ['E'],
        'I': ['Y', 'E'],
        'SK': ['S', 'SCH']
    }
    
    # Generate substitution variants
    for old, replacements in substitutions.items():
        if old in word:
            for new in replacements:
                variations.add(word.replace(old, new))
    
    # Common Dutch/English variations and typos
    common_variations = {
        'WHISKEY': ['WHISKY', 'WISKY', 'WISKEE', 'WHISKIE', 'WHISKEE'],
        'WHISKY': ['WHISKEY', 'WISKY', 'WISKEE', 'WHISKIE', 'WHISKEE'],  
        'WISKY': ['WHISKEY', 'WHISKY', 'WISKEE', 'WHISKIE', 'WHISKEE'],
        'KERK': ['CHURCH', 'KARK', 'KIRK'],
        'CHURCH': ['KERK', 'CHIRCH', 'CHUCH'],
        'DOKTER': ['DOCTOR', 'DOCTER', 'DOKTOR'],
        'DOCTOR': ['DOKTER', 'DOCTER', 'DOKTOR']
    }
    
    # Add the word itself to check for its variations
    for base_word, variants in common_variations.items():
        if word in variants or word == base_word:
            variations.update(variants)
            variations.add(base_word)
    
    # Simple character-level edit distance variations (add/remove/substitute one character)
    if len(word) > 3:  # Only for longer words to avoid too many false matches
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # Single character substitutions
        for i in range(len(word)):
            for char in alphabet:
                if char != word[i]:
                    variant = word[:i] + char + word[i+1:]
                    variations.add(variant)
        
        # Single character deletions
        for i in range(len(word)):
            variant = word[:i] + word[i+1:]
            if len(variant) > 2:  # Keep reasonable length
                variations.add(variant)
        
        # Single character insertions (only common ones to avoid explosion)
        common_inserts = ['S', 'E', 'H']
        for i in range(len(word) + 1):
            for char in common_inserts:
                variant = word[:i] + char + word[i:]
                if len(variant) <= len(word) + 2:  # Reasonable length limit
                    variations.add(variant)
    
    return list(variations)


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
    """Create training data from the emoji dictionary with controlled fuzzy variations."""
    emoji_dict = load_emoji_dict()
    training_data = []
    
    for word, emoji in emoji_dict.items():
        if emoji != "❓":  # Skip default emoji entries
            training_data.append({"summary": word, "emoji": emoji})
            
            # Add only the most important fuzzy variations to avoid bias
            if word == 'WHISKEY':
                # Only add the most common typos/variations
                important_variants = ['WHISKY', 'WISKY']
                for variant in important_variants:
                    training_data.append({"summary": variant, "emoji": emoji})
            elif word == 'KERK':
                training_data.append({"summary": 'CHURCH', "emoji": emoji})
            elif word == 'DOKTER':
                training_data.append({"summary": 'DOCTOR', "emoji": emoji})
    
    print(f"\nFound {len(training_data)} entries in emoji dictionary (including key variations, excluding ❓)")
    return training_data


def augment_training_data(word_emoji_pairs):
    """Augment training data by creating variations and combinations with better balance."""
    augmented = []
    emoji_counts = Counter([item["emoji"] for item in word_emoji_pairs])
    
    # Calculate balance factor - reduce augmentation for over-represented emojis
    max_count = max(emoji_counts.values())
    balance_factors = {emoji: max(1, max_count // count) for emoji, count in emoji_counts.items()}
    
    for item in word_emoji_pairs:
        text = item["summary"]
        emoji = item["emoji"]
        balance_factor = min(balance_factors[emoji], 3)  # Cap at 3x augmentation
        
        # Add original
        augmented.append(item)
        
        # Only add fuzzy variations for short, specific words to avoid over-augmentation
        words = text.split()
        if len(words) == 1 and len(words[0]) > 4 and balance_factor > 1:
            # Generate limited fuzzy variations for single important words
            word_variations = create_fuzzy_matches(words[0])[:3]  # Limit to 3 variations
            for variation in word_variations:
                if variation != words[0] and len(variation) > 3:
                    augmented.append({"summary": variation, "emoji": emoji})
        
        # Add contextual variations (also limited by balance factor)
        contextual_variations = [
            f"NAAR {text}",
            f"{text} DOEN", 
            f"{text} AVOND",
            f"AFSPRAAK {text}",
            f"{text} BEZOEKEN",
            f"{text} PLANNEN"
        ]
        
        for i, variation in enumerate(contextual_variations[:balance_factor]):
            if variation != text:
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
    Train an ensemble of classifiers with class balancing for better predictions.
    
    Args:
        X_train: List of text phrases
        y_train: List of corresponding emojis
        
    Returns:
        Dictionary of trained models
    """
    print("\n🔧 Training balanced classifiers...")
    
    # Use 'balanced' mode for class weights to handle imbalance
    print(f"   Training data has {len(set(y_train))} unique classes")
    
    # Create different feature extraction approaches with better parameters for fuzzy matching
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer='char_wb',  # Character n-grams to handle typos better
        ngram_range=(2, 6) if len(''.join(X_train)) > 1000 else (2, 4),
        max_features=800,    # More features for better coverage
        min_df=1,
        max_df=0.90
    )
    
    # Word-level TF-IDF for semantic understanding
    word_tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 3),
        max_features=500,
        min_df=1,
        max_df=0.95
    )
    
    count_vectorizer = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=400,
        min_df=1
    )
    
    # Individual classifiers with class balancing
    models = {}
    
    # Balanced Logistic Regression with character-level TF-IDF (better for typos)
    lr_char = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('classifier', LogisticRegression(
            C=0.5, 
            max_iter=2000, 
            random_state=42,
            class_weight='balanced',
            solver='saga'  # Better for many classes and balanced weights
        ))
    ])
    
    # Word-level Logistic Regression
    lr_word = Pipeline([
        ('tfidf', word_tfidf),
        ('classifier', LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # Naive Bayes (naturally handles imbalance well)
    nb_count = Pipeline([
        ('count', count_vectorizer),
        ('classifier', MultinomialNB(alpha=0.5))  # Lower alpha for less smoothing
    ])
    
    # Balanced Random Forest
    rf_balanced = Pipeline([
        ('tfidf', word_tfidf),
        ('classifier', RandomForestClassifier(
            n_estimators=150, 
            random_state=42, 
            max_depth=15,
            class_weight='balanced',
            min_samples_split=3,
            min_samples_leaf=1
        ))
    ])
    
    # Train individual models
    models['logistic_char'] = lr_char
    models['logistic_word'] = lr_word
    models['naive_bayes'] = nb_count  
    models['random_forest'] = rf_balanced
    
    for name, model in models.items():
        print(f"   Training {name}...")
        try:
            model.fit(X_train, y_train)
            
            # Quick cross-validation score
            if len(set(y_train)) > 1:
                scores = cross_val_score(model, X_train, y_train, cv=min(3, len(X_train)//5), scoring='f1_weighted')
                print(f"   {name} CV F1 score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        except Exception as e:
            print(f"   Failed to train {name}: {e}")
            continue
    
    # Create balanced ensemble
    ensemble_estimators = [(name, model) for name, model in models.items() 
                          if name != 'ensemble']
    
    if len(ensemble_estimators) > 1:
        ensemble = VotingClassifier(
            ensemble_estimators,
            voting='soft'
        )
        
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
    
    def _edit_distance(self, s1, s2):
        """Calculate simple edit distance between two strings."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
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
        1. Direct and fuzzy word lookup in emoji dictionary
        2. ML model prediction with confidence scoring
        3. Intelligent combination with bias mitigation
        """
        text_clean = sanitize_text(text)
        words = text_clean.split()
        
        # Strategy 1: Direct and fuzzy dictionary lookup
        dict_matches = []
        fuzzy_matches = []
        
        for word in words:
            # Direct match
            if word.lower() in self.word_to_emoji:
                dict_matches.append((word, self.word_to_emoji[word.lower()], 1.0))
            else:
                # Fuzzy match using edit distance for close matches
                best_fuzzy_match = None
                best_confidence = 0
                
                for dict_word in self.word_to_emoji.keys():
                    if abs(len(word) - len(dict_word)) <= 2:  # Similar length
                        # Calculate simple edit distance
                        distance = self._edit_distance(word.lower(), dict_word)
                        max_len = max(len(word), len(dict_word))
                        
                        if distance <= 2 and max_len > 3:  # Allow 1-2 character differences for words > 3 chars
                            confidence = 1.0 - (distance / max_len)
                            if confidence > best_confidence and confidence > 0.6:
                                best_confidence = confidence
                                best_fuzzy_match = dict_word
                
                if best_fuzzy_match:
                    fuzzy_matches.append((word, self.word_to_emoji[best_fuzzy_match], best_confidence))
        
        # Strategy 2: ML prediction with ensemble voting
        ml_predictions = {}
        best_ml_prediction = None
        best_ml_confidence = 0
        
        if self.models and len(text_clean.strip()) > 0:
            try:
                # Get predictions from all models
                for name, model in self.models.items():
                    if name == 'ensemble':
                        continue
                    try:
                        prediction = model.predict([text_clean])[0]
                        probabilities = model.predict_proba([text_clean])[0]
                        max_prob = np.max(probabilities)
                        
                        ml_predictions[name] = {
                            'prediction': prediction,
                            'confidence': max_prob
                        }
                    except Exception as e:
                        logger.debug(f"Model {name} prediction failed: {e}")
                
                # Ensemble prediction if available
                if 'ensemble' in self.models:
                    try:
                        ensemble = self.models['ensemble']
                        best_ml_prediction = ensemble.predict([text_clean])[0]
                        proba = ensemble.predict_proba([text_clean])[0]
                        best_ml_confidence = np.max(proba)
                    except:
                        pass
                
                # If no ensemble, use voting from individual models
                if not best_ml_prediction and ml_predictions:
                    # Weighted voting based on confidence
                    emoji_votes = defaultdict(float)
                    total_weight = 0
                    
                    for model_name, pred_info in ml_predictions.items():
                        weight = pred_info['confidence']
                        emoji_votes[pred_info['prediction']] += weight
                        total_weight += weight
                    
                    if emoji_votes:
                        best_ml_prediction = max(emoji_votes.keys(), key=lambda x: emoji_votes[x])
                        best_ml_confidence = emoji_votes[best_ml_prediction] / total_weight if total_weight > 0 else 0
                        
            except Exception as e:
                logger.debug(f"ML prediction failed: {e}")
        
        # Strategy 3: Intelligent combination with bias mitigation
        all_matches = dict_matches + fuzzy_matches
        
        # Bias detection: check if ML prediction is over-represented emoji
        over_represented = ['⛪', '🦎', '🚗']  # Top biased emojis from training
        ml_is_biased = best_ml_prediction in over_represented
        
        if all_matches and best_ml_prediction:
            # Get all dictionary emoji candidates
            dict_emojis = [emoji for _, emoji, _ in all_matches]
            dict_confidences = [conf for _, _, conf in all_matches]
            max_dict_confidence = max(dict_confidences) if dict_confidences else 0
            
            # If ML confirms dictionary match
            if best_ml_prediction in dict_emojis:
                return best_ml_prediction
            
            # If dictionary has high confidence match, prefer it over biased ML
            elif max_dict_confidence >= 0.8 and ml_is_biased and best_ml_confidence < 0.8:
                best_dict_match = max(all_matches, key=lambda x: x[2])
                return best_dict_match[1]
            
            # If ML has very high confidence and is not biased
            elif best_ml_confidence > 0.8 and not ml_is_biased:
                return best_ml_prediction
            
            # Default to dictionary for ambiguous cases
            else:
                best_dict_match = max(all_matches, key=lambda x: x[2])
                return best_dict_match[1]
        
        elif all_matches:
            # Only dictionary matches
            best_dict_match = max(all_matches, key=lambda x: x[2])
            return best_dict_match[1]
        
        elif best_ml_prediction:
            # Only ML prediction - apply bias penalty
            if ml_is_biased and best_ml_confidence < 0.6:
                return "❓"  # Too uncertain and biased
            else:
                return best_ml_prediction
        
        else:
            return "❓"  # No confident prediction
    
    def predict_with_probabilities(self, text):
        """Get prediction with detailed probability information."""
        text_clean = sanitize_text(text)
        
        result = {
            'prediction': self.predict_emoji(text),
            'dictionary_matches': [],
            'ml_predictions': {},
            'confidence': 0
        }
        
        # Check dictionary matches (both direct and fuzzy)
        words = text_clean.split()
        for word in words:
            # Direct match
            if word.lower() in self.word_to_emoji:
                result['dictionary_matches'].append({
                    'word': word,
                    'emoji': self.word_to_emoji[word.lower()],
                    'confidence': 1.0,
                    'type': 'exact'
                })
            else:
                # Fuzzy match using edit distance
                best_fuzzy_match = None
                best_confidence = 0
                
                for dict_word in self.word_to_emoji.keys():
                    if abs(len(word) - len(dict_word)) <= 2:  # Similar length
                        distance = self._edit_distance(word.lower(), dict_word)
                        max_len = max(len(word), len(dict_word))
                        min_len = min(len(word), len(dict_word))
                        
                        # Require minimum length and reasonable similarity
                        if distance <= 2 and min_len >= 5 and max_len > 4:  # Stricter length requirements
                            confidence = 1.0 - (distance / max_len)
                            # Higher threshold and check for similar starting characters
                            if confidence > best_confidence and confidence > 0.70:
                                # Additional check: first 2 characters should be similar for very close matches
                                first_chars_match = word[0].lower() == dict_word[0].lower()
                                if confidence > 0.80 or first_chars_match:
                                    best_confidence = confidence
                                    best_fuzzy_match = dict_word
                
                if best_fuzzy_match:
                    result['dictionary_matches'].append({
                        'word': word,
                        'emoji': self.word_to_emoji[best_fuzzy_match],
                        'confidence': best_confidence,
                        'type': 'fuzzy',
                        'matched_word': best_fuzzy_match
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