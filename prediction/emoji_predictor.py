"""
Emoji Predictor for CalDAV-Emojifier
====================================
Production-ready emoji prediction combining ML models with dictionary lookup.
"""

import json
import os
import logging
import re
import pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from collections import Counter

logger = logging.getLogger(__name__)


def is_emoji(character):
    """Check if character is an emoji."""
    character = character.encode("unicode-escape").decode("utf-8")
    return (character.startswith("\\U") or character.startswith("\\u"))


def sanitize_text(text):
    """Clean and normalize text for consistent processing."""
    if not text:
        return ""
    sanitized = ''.join(char for char in text if char.isalpha() or char.isspace())
    return sanitized.upper().strip()


class EmojiPredictor:
    """Production emoji predictor with caching and persistence."""
    
    def __init__(self, config_path="../config", cache_models=True):
        self.config_path = Path(config_path)
        self.cache_models = cache_models
        self.models = None
        self.emoji_dict = self._load_emoji_dict()
        self.word_to_emoji = self._create_word_mapping()
        self.model_cache_file = self.config_path / "trained_models.pkl"
        
        # Try to load cached models
        if self.cache_models and self.model_cache_file.exists():
            try:
                self._load_cached_models()
                logger.info("Loaded cached emoji prediction models")
            except Exception as e:
                logger.warning(f"Failed to load cached models: {e}")
    
    def _load_emoji_dict(self):
        """Load the existing emoji dictionary."""
        emoji_dict_path = self.config_path / "emoji_dict.json"
        if emoji_dict_path.exists():
            with open(emoji_dict_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _create_word_mapping(self):
        """Create direct word-to-emoji mapping for fast lookup."""
        return {word.lower(): emoji for word, emoji in self.emoji_dict.items() 
                if emoji != "❓"}
    
    def _parse_training_data(self):
        """Parse training.ics file to extract emoji-text pairs."""
        training_file = Path(__file__).parent / "training.ics"
        if not training_file.exists():
            logger.warning("training.ics not found")
            return []
        
        with open(training_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract SUMMARY lines
        summary_pattern = r'SUMMARY:(.+)'
        summaries = re.findall(summary_pattern, content)
        
        emoji_summaries = []
        seen_combinations = set()
        
        for summary in summaries:
            cleaned = summary.strip()
            if cleaned and is_emoji(cleaned[0]):
                emoji = cleaned[0]
                if emoji == "❓":
                    continue
                    
                text_without_emoji = sanitize_text(cleaned[1:])
                if text_without_emoji:
                    combination_key = (text_without_emoji, emoji)
                    if combination_key not in seen_combinations:
                        seen_combinations.add(combination_key)
                        emoji_summaries.append({
                            "summary": text_without_emoji, 
                            "emoji": emoji
                        })
        
        return emoji_summaries
    
    def _create_training_data(self):
        """Create comprehensive training dataset."""
        # Get calendar training data
        calendar_data = self._parse_training_data()
        
        # Get dictionary data
        dict_data = []
        for word, emoji in self.emoji_dict.items():
            if emoji != "❓":
                dict_data.append({"summary": word, "emoji": emoji})
        
        # Combine datasets
        all_data = calendar_data + dict_data
        
        # Add variations for better generalization
        augmented_data = []
        for item in all_data:
            text = item["summary"]
            emoji = item["emoji"]
            
            # Add original
            augmented_data.append(item)
            
            # Add variations
            variations = [
                f"NAAR {text}",
                f"{text} DOEN",
                f"{text} AFSPRAAK",
                f"BEZOEKEN {text}",
                f"{text} PLANNEN"
            ]
            
            for variation in variations:
                augmented_data.append({"summary": variation, "emoji": emoji})
        
        # Deduplicate
        seen = set()
        deduplicated = []
        for item in augmented_data:
            key = (item["summary"], item["emoji"])
            if key not in seen:
                seen.add(key)
                deduplicated.append(item)
        
        X = [item["summary"] for item in deduplicated]
        y = [item["emoji"] for item in deduplicated]
        
        logger.info(f"Created training dataset: {len(X)} samples, {len(set(y))} unique emojis")
        return X, y
    
    def train(self, force_retrain=False):
        """Train the prediction models."""
        if self.models is not None and not force_retrain:
            return
        
        X_train, y_train = self._create_training_data()
        
        if len(X_train) < 10:  # Not enough data
            logger.warning("Insufficient training data, using dictionary-only mode")
            return
        
        logger.info("Training emoji prediction models...")
        
        # Create ensemble of classifiers
        models = {}
        
        # Logistic Regression with TF-IDF
        lr_tfidf = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 3),
                max_features=500,
                min_df=1,
                max_df=0.95
            )),
            ('classifier', LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            ))
        ])
        
        # Naive Bayes with Count Vectorizer
        nb_count = Pipeline([
            ('count', CountVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=300,
                min_df=1
            )),
            ('classifier', MultinomialNB(alpha=1.0))
        ])
        
        # Random Forest
        rf_tfidf = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=200
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                max_depth=8
            ))
        ])
        
        # Train individual models
        models['logistic'] = lr_tfidf
        models['naive_bayes'] = nb_count
        models['random_forest'] = rf_tfidf
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            logger.debug(f"Trained {name} model")
        
        # Create ensemble
        ensemble = VotingClassifier([
            ('lr', lr_tfidf),
            ('nb', nb_count),
            ('rf', rf_tfidf)
        ], voting='soft')
        
        ensemble.fit(X_train, y_train)
        models['ensemble'] = ensemble
        
        self.models = models
        logger.info("Model training completed")
        
        # Cache models if enabled
        if self.cache_models:
            self._save_cached_models()
    
    def _save_cached_models(self):
        """Save trained models to cache file."""
        try:
            os.makedirs(self.config_path, exist_ok=True)
            with open(self.model_cache_file, 'wb') as f:
                pickle.dump(self.models, f)
            logger.debug("Saved models to cache")
        except Exception as e:
            logger.warning(f"Failed to cache models: {e}")
    
    def _load_cached_models(self):
        """Load models from cache file."""
        with open(self.model_cache_file, 'rb') as f:
            self.models = pickle.load(f)
    
    def predict_emoji(self, text, fallback_emoji="❓"):
        """
        Predict the best emoji for given text.
        
        Args:
            text: Input text to predict emoji for
            fallback_emoji: Emoji to return if no good prediction found
            
        Returns:
            str: Predicted emoji
        """
        if not text or not text.strip():
            return fallback_emoji
        
        text_clean = sanitize_text(text)
        words = text_clean.split()
        
        # Strategy 1: Direct dictionary lookup (highest priority)
        for word in words:
            if word.lower() in self.word_to_emoji:
                emoji = self.word_to_emoji[word.lower()]
                logger.debug(f"Dictionary match: '{word}' -> {emoji}")
                return emoji
        
        # Strategy 2: ML prediction (if models available)
        if self.models and len(text_clean) > 0:
            try:
                ensemble = self.models['ensemble']
                ml_prediction = ensemble.predict([text_clean])[0]
                
                # Get confidence from multiple models
                confidences = []
                for name, model in self.models.items():
                    if name != 'ensemble':
                        try:
                            proba = model.predict_proba([text_clean])[0]
                            pred_classes = model.classes_
                            if ml_prediction in pred_classes:
                                pred_idx = list(pred_classes).index(ml_prediction)
                                confidences.append(proba[pred_idx])
                        except:
                            pass
                
                avg_confidence = np.mean(confidences) if confidences else 0
                
                # Only return ML prediction if confidence is reasonable
                if avg_confidence > 0.1:  # 10% threshold
                    logger.debug(f"ML prediction: '{text_clean}' -> {ml_prediction} (conf: {avg_confidence:.2f})")
                    return ml_prediction
                
            except Exception as e:
                logger.debug(f"ML prediction failed: {e}")
        
        # Strategy 3: Partial word matching in dictionary
        for word in words:
            for dict_word in self.word_to_emoji:
                if word.lower() in dict_word or dict_word in word.lower():
                    emoji = self.word_to_emoji[dict_word]
                    logger.debug(f"Partial match: '{word}' matches '{dict_word}' -> {emoji}")
                    return emoji
        
        logger.debug(f"No prediction found for '{text}', using fallback")
        return fallback_emoji
    
    def get_prediction_info(self, text):
        """Get detailed prediction information for debugging."""
        text_clean = sanitize_text(text)
        words = text_clean.split()
        
        info = {
            'text': text,
            'cleaned_text': text_clean,
            'words': words,
            'final_prediction': self.predict_emoji(text),
            'dictionary_matches': [],
            'ml_predictions': {},
            'partial_matches': []
        }
        
        # Check dictionary matches
        for word in words:
            if word.lower() in self.word_to_emoji:
                info['dictionary_matches'].append({
                    'word': word,
                    'emoji': self.word_to_emoji[word.lower()]
                })
        
        # Check partial matches
        for word in words:
            for dict_word in self.word_to_emoji:
                if word.lower() != dict_word and (word.lower() in dict_word or dict_word in word.lower()):
                    info['partial_matches'].append({
                        'word': word,
                        'matched_dict_word': dict_word,
                        'emoji': self.word_to_emoji[dict_word]
                    })
        
        # Get ML predictions
        if self.models and len(text_clean) > 0:
            for name, model in self.models.items():
                try:
                    prediction = model.predict([text_clean])[0]
                    probabilities = model.predict_proba([text_clean])[0]
                    classes = model.classes_
                    
                    prob_dict = {emoji: prob for emoji, prob in zip(classes, probabilities)}
                    max_prob = max(prob_dict.values())
                    
                    info['ml_predictions'][name] = {
                        'prediction': prediction,
                        'confidence': max_prob,
                        'top_3': sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                    }
                except Exception as e:
                    info['ml_predictions'][name] = {'error': str(e)}
        
        return info


# Convenience function for easy integration
def create_emoji_predictor(config_path="../config", auto_train=True):
    """
    Create and optionally train an emoji predictor.
    
    Args:
        config_path: Path to config directory
        auto_train: Whether to train models automatically
        
    Returns:
        EmojiPredictor: Ready-to-use predictor
    """
    predictor = EmojiPredictor(config_path)
    
    if auto_train:
        predictor.train()
    
    return predictor


# Simple function for direct integration into sync.py
def predict_emoji_for_text(text, config_path="../config"):
    """
    Simple function to predict emoji for text. Handles model loading/training automatically.
    
    Args:
        text: Text to predict emoji for
        config_path: Path to config directory
        
    Returns:
        str: Predicted emoji
    """
    predictor = create_emoji_predictor(config_path, auto_train=True)
    return predictor.predict_emoji(text)


if __name__ == "__main__":
    # Demo usage
    predictor = create_emoji_predictor()
    
    test_phrases = [
        "TANDARTS AFSPRAAK",
        "BORREL MET VRIENDEN", 
        "WERKEN IN ARNHEM",
        "VAKANTIE PLANNEN",
        "KERK DIENST",
        "SPELLETJESAVOND",
        "HUISARTS BEZOEK",
        "BOODSCHAPPEN DOEN"
    ]
    
    print("🎯 Emoji Predictor Demo")
    print("=" * 40)
    
    for phrase in test_phrases:
        emoji = predictor.predict_emoji(phrase)
        print(f"{phrase:25} → {emoji}")