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
