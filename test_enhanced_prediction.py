#!/usr/bin/env python3
"""
Test the enhanced emoji prediction integration with sync.py
"""

import sys
import os
sys.path.append('.')
sys.path.append('./llm')

# Import the functions from sync.py
from sync import words_to_emoji, add_words_to_emoji_dict, sanitize_word

def test_enhanced_prediction():
    """Test the enhanced emoji prediction functionality."""
    
    print("🧪 Testing Enhanced Emoji Prediction Integration")
    print("=" * 50)
    
    test_cases = [
        # (input_words, expected_emoji_type)
        (["TANDARTS", "AFSPRAAK"], "medical"),
        (["BORREL", "VRIENDEN"], "social"),  
        (["VAKANTIE", "PLANNEN"], "vacation"),
        (["WERK", "VERGADERING"], "work"),
        (["KAPPER", "BEZOEK"], "personal"),
        (["SPELLETJESAVOND"], "entertainment"),
        (["HUISARTS"], "medical"),
        (["NIEUWE", "WOORDEN", "TEST"], "unknown")  # Should get predicted emoji
    ]
    
    print("\n📝 Testing individual predictions:")
    for words, category in test_cases:
        print(f"\nInput: {words}")
        emoji = words_to_emoji(words)
        print(f"Predicted: {emoji} (category: {category})")
        
        # Test the add function as well
        test_word = ["TEST", "WORD"] + words[:1]  # New word combination
        add_words_to_emoji_dict(test_word, "❓")
        
    print(f"\n✅ All tests completed!")
    
    # Show some stats
    try:
        from emoji_predictor import EmojiPredictor
        predictor = EmojiPredictor(config_path="./config")
        
        print(f"\n📊 Predictor Stats:")
        print(f"   Dictionary entries: {len(predictor.emoji_dict)}")
        print(f"   Active mappings: {len(predictor.word_to_emoji)}")
        
        # Train if not already done
        if predictor.models is None:
            print("   🔧 Training models...")
            predictor.train()
        
        if predictor.models:
            print(f"   Trained models: {len(predictor.models)}")
        
    except Exception as e:
        print(f"   ⚠️ Could not load predictor stats: {e}")

if __name__ == "__main__":
    test_enhanced_prediction()