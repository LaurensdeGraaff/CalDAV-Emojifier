import unittest
from unittest.mock import patch, MagicMock
import sync
from sync import sanitize_word, is_emoji, update_or_add_word_to_emoji_dict, add_words_to_emoji_dict, words_to_emoji

class TestSync(unittest.TestCase):

    def test_sanitize_word(self):
        self.assertEqual(sanitize_word("Hello!"), "HELLO")
        self.assertEqual(sanitize_word("123@#$"), "")
        self.assertEqual(sanitize_word("Hello World!"), "HELLOWORLD")
        self.assertEqual(sanitize_word("🍰 the cake is a lie"), "THECAKEISALIE")
        self.assertEqual(sanitize_word("The 🍰 is a lie"), "THEISALIE")
        self.assertEqual(sanitize_word(""), "")

    def test_is_emoji(self):
        self.assertTrue(is_emoji("😊"))
        self.assertTrue(is_emoji("😊word"))
        self.assertTrue(is_emoji("🎉"))
        self.assertFalse(is_emoji("A"))
        self.assertFalse(is_emoji("1"))
        self.assertFalse(is_emoji("?"))
        self.assertFalse(is_emoji("?🍰"))

    @patch("sync.json.dump")
    @patch("sync.open", create=True)
    def test_update_or_add_word_to_emoji_dict(self, mock_open, mock_json_dump): 
        with patch("sync.emoji_dict", {"HELLO": "😊"}):
            self.assertEqual(sync.emoji_dict["HELLO"], "😊")
            result = update_or_add_word_to_emoji_dict("HELLO", "🎉")
            self.assertTrue(result)
            self.assertEqual(sync.emoji_dict["HELLO"], "🎉")

            result = update_or_add_word_to_emoji_dict("WORLD", "🌍")
            self.assertTrue(result)
            self.assertEqual(sync.emoji_dict["WORLD"], "🌍")

    @patch("sync.json.dump")
    @patch("sync.open", create=True)
    def test_add_words_to_emoji_dict(self, mock_open, mock_json_dump): 
        with patch("sync.emoji_dict", {"HELLO": "😊"}):
            words_from_event = "GET CAKE FROM GLADOS".split(" ")
            add_words_to_emoji_dict(words_from_event)
            self.assertEqual(sync.emoji_dict["GET"], "❓")
            self.assertNotIn("CAKE", sync.emoji_dict)
            self.assertNotIn("FROM", sync.emoji_dict)
            self.assertNotIn("GLADOS", sync.emoji_dict)
            self.assertEqual(sync.emoji_dict["HELLO"], "😊")
    
    @patch("sync.json.dump")
    @patch("sync.open", create=True)
    def test_words_to_emoji(self, mock_open, mock_json_dump): 
        with patch("sync.emoji_dict", {"HELLO": "😊", "CAKE":"🍰"}):
            words_from_event = "GET CAKE FROM GLADOS".split(" ")
            result = words_to_emoji(words_from_event)
            self.assertEqual(result, "🍰")  
            words_from_event = "HELLO NEIGHBOUR".split(" ")
            result = words_to_emoji(words_from_event)
            self.assertEqual(result, "😊")  
            words_from_event = "THIS IS NEW".split(" ")
            result = words_to_emoji(words_from_event)
            self.assertEqual(result, "❓")  

if __name__ == "__main__":
    unittest.main()