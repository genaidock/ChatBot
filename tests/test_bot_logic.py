import sys
import unittest
from unittest.mock import MagicMock, AsyncMock

# Provide mock modules if not installed in local development environment
for mod in ["httpx", "dotenv", "openai"]:
    if mod not in sys.modules:
        mock_m = MagicMock()
        mock_m.load_dotenv = MagicMock()
        sys.modules[mod] = mock_m

try:
    import telegram
    from telegram import InlineKeyboardMarkup, InlineKeyboardButton
except ImportError:
    # Create lightweight mock classes
    class InlineKeyboardButton:
        def __init__(self, text, callback_data=None, switch_inline_query_current_chat=None):
            self.text = text
            self.callback_data = callback_data
            self.switch_inline_query_current_chat = switch_inline_query_current_chat

    class InlineKeyboardMarkup:
        def __init__(self, inline_keyboard):
            self.inline_keyboard = inline_keyboard

    class BotCommand:
        def __init__(self, command, description):
            self.command = command
            self.description = description

    class Constants:
        class ChatAction:
            TYPING = "typing"
            UPLOAD_PHOTO = "upload_photo"

    class BadRequest(Exception):
        pass

    class RetryAfter(Exception):
        pass

    class TelegramError(Exception):
        pass

    mock_telegram = MagicMock()
    mock_telegram.InlineKeyboardMarkup = InlineKeyboardMarkup
    mock_telegram.InlineKeyboardButton = InlineKeyboardButton
    mock_telegram.BotCommand = BotCommand
    mock_telegram.constants = Constants
    mock_telegram.error = MagicMock()
    mock_telegram.error.BadRequest = BadRequest
    mock_telegram.error.RetryAfter = RetryAfter
    mock_telegram.error.TelegramError = TelegramError

    sys.modules["telegram"] = mock_telegram
    sys.modules["telegram.error"] = mock_telegram.error
    sys.modules["telegram.ext"] = MagicMock()
    sys.modules["telegram.constants"] = Constants

# Now import bot module
import bot

class TestBotLogic(unittest.TestCase):

    def setUp(self):
        # Reset internal stores
        bot._CHAT_HISTORY.clear()
        bot._PERSONAS.clear()
        bot._CHAT_MODELS.clear()

    def test_chat_config_defaults(self):
        cfg = bot.get_chat_config(12345)
        self.assertIn("local_model", cfg)
        self.assertIn("omniroute_model", cfg)
        self.assertIn("priority", cfg)

    def test_chat_config_with_context(self):
        mock_context = MagicMock()
        mock_context.chat_data = {}

        cfg = bot.get_chat_config(12345, mock_context)
        self.assertIn("model_config", mock_context.chat_data)
        self.assertEqual(cfg, mock_context.chat_data["model_config"])

        # Update config
        cfg["omniroute_model"] = "auto/deepseek"
        bot.set_chat_config(12345, cfg, mock_context)
        self.assertEqual(mock_context.chat_data["model_config"]["omniroute_model"], "auto/deepseek")

    def test_persona_management(self):
        mock_context = MagicMock()
        mock_context.chat_data = {}

        # Default persona
        p = bot.get_persona(999, mock_context)
        self.assertEqual(p, bot.DEFAULT_PERSONA)

        # Set custom persona
        bot.set_persona(999, "You are a senior python expert.", mock_context)
        self.assertEqual(bot.get_persona(999, mock_context), "You are a senior python expert.")

        # Reset persona
        bot.reset_persona(999, mock_context)
        self.assertEqual(bot.get_persona(999, mock_context), bot.DEFAULT_PERSONA)

    def test_chat_history_trimming(self):
        mock_context = MagicMock()
        mock_context.chat_data = {}

        # Add 20 messages
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        bot.set_chat_history(888, history, mock_context)

        stored = bot.get_chat_history(888, mock_context)
        self.assertEqual(len(stored), 12, "Chat history should be capped at 12 messages")
        self.assertEqual(stored[-1]["content"], "msg 19")

        # Clear history
        bot.clear_chat_history(888, mock_context)
        self.assertEqual(len(bot.get_chat_history(888, mock_context)), 0)

    def test_build_model_keyboard(self):
        cfg = {
            "local_model": "dolphin3:latest",
            "omniroute_model": "auto/deepseek",
            "priority": "local"
        }
        kb = bot.build_model_keyboard(cfg)
        self.assertIsNotNone(kb)
        
        # Flatten buttons
        buttons = [b for row in kb.inline_keyboard for b in row]
        button_texts = [b.text for b in buttons]
        callback_data = [b.callback_data for b in buttons]

        # Verify active checkmark on selected local model
        self.assertTrue(any("✅" in txt and "Dolphin 3.0" in txt for txt in button_texts))
        self.assertIn("model:dolphin", callback_data)
        self.assertIn("model:deepseek", callback_data)
        self.assertIn("model:toggle_priority", callback_data)
        self.assertIn("act:ollama", callback_data)
        self.assertIn("act:clear", callback_data)

    def test_build_start_keyboard(self):
        kb = bot.build_start_keyboard()
        buttons = [b for row in kb.inline_keyboard for b in row]
        callback_data = [b.callback_data for b in buttons if b.callback_data]
        switch_inline = [b.switch_inline_query_current_chat for b in buttons if b.switch_inline_query_current_chat]
        self.assertIn("act:open_model", callback_data)
        self.assertIn("act:ollama", callback_data)
        self.assertIn("act:clear", callback_data)
        self.assertIn("/imagine ", switch_inline)
        self.assertIn("/edit ", switch_inline)
        self.assertIn("/search ", switch_inline)
        self.assertIn("/speak ", switch_inline)

    def test_clean_text_for_speech(self):
        raw = "Check [Google](https://google.com) and **bold** *italic* text with `code`.\n```python\nprint(123)\n```"
        cleaned = bot.clean_text_for_speech(raw)
        self.assertNotIn("https://", cleaned)
        self.assertIn("Google", cleaned)
        self.assertIn("bold italic", cleaned)
        self.assertIn("[code snippet omitted]", cleaned)
        self.assertNotIn("`", cleaned)

        # Truncation test
        long_text = "This is a sentence. " * 100
        truncated = bot.clean_text_for_speech(long_text, max_chars=100)
        self.assertTrue(len(truncated) <= 120)
        self.assertIn("truncated", truncated)

    def test_parse_speak_args(self):
        # 1. Default voice
        text, voice = bot.parse_speak_args(["Hello", "world"])
        self.assertEqual(text, "Hello world")
        self.assertEqual(voice, "af_heart")

        # 2. First arg voice alias
        text, voice = bot.parse_speak_args(["puck", "Hello", "world"])
        self.assertEqual(text, "Hello world")
        self.assertEqual(voice, "am_puck")

        # 3. Flag argument
        text, voice = bot.parse_speak_args(["--voice", "isabella", "Good", "morning"])
        self.assertEqual(text, "Good morning")
        self.assertEqual(voice, "bf_isabella")

        text, voice = bot.parse_speak_args(["--voice=fable", "Welcome", "back"])
        self.assertEqual(text, "Welcome back")
        self.assertEqual(voice, "bm_fable")

        # 4. Reply to text message without args
        mock_reply = MagicMock()
        mock_reply.text = "This is a reply to an AI response."
        mock_reply.caption = None
        text, voice = bot.parse_speak_args([], reply_to_message=mock_reply)
        self.assertEqual(text, "This is a reply to an AI response.")
        self.assertEqual(voice, "af_heart")

        # 5. Reply to text message with voice arg
        text, voice = bot.parse_speak_args(["puck"], reply_to_message=mock_reply)
        self.assertEqual(text, "This is a reply to an AI response.")
        self.assertEqual(voice, "am_puck")

        # 6. Reply to photo caption
        mock_reply_cap = MagicMock()
        mock_reply_cap.text = None
        mock_reply_cap.caption = "Photo caption text"
        text, voice = bot.parse_speak_args([], reply_to_message=mock_reply_cap)
        self.assertEqual(text, "Photo caption text")

    async def _async_speak_cmd_help_test(self):
        mock_update = MagicMock()
        mock_update.effective_chat.id = 12345
        mock_update.message.reply_to_message = None
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock()
        mock_context.args = []

        await bot.speak_cmd(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once()
        call_text = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Paradee Text-to-Speech", call_text)
        self.assertIn("/speak", call_text)

    def test_speak_cmd_help_when_empty(self):
        import asyncio
        asyncio.run(self._async_speak_cmd_help_test())

    async def _async_safe_edit_test(self):
        # 1. Normal edit success
        target = MagicMock()
        target.edit_text = AsyncMock(return_value=True)
        res = await bot.safe_edit_message(target, "Hello world", parse_mode="HTML")
        self.assertTrue(res)
        target.edit_text.assert_called_with(text="Hello world", reply_markup=None, parse_mode="HTML")

        # 2. Parsing error falls back to plain text
        target2 = MagicMock()
        target2.edit_text = AsyncMock(side_effect=[
            bot.BadRequest("Can't parse entities: can't find end of tag"),
            True
        ])
        res2 = await bot.safe_edit_message(target2, "Broken <tag> text", parse_mode="HTML")
        self.assertTrue(res2)
        self.assertEqual(target2.edit_text.call_count, 2)
        # Second call should have parse_mode=None
        self.assertIsNone(target2.edit_text.call_args[1]["parse_mode"])

    def test_safe_edit_message(self):
        import asyncio
        asyncio.run(self._async_safe_edit_test())

if __name__ == "__main__":
    unittest.main()
