import os
import asyncio
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID")
WEBHOOK_URL = os.getenv("WEBHOOK_URL") # e.g. https://your-app-name.onrender.com

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3.5-flash-lite") # Use 3.5-flash-lite for higher limits for new users

# Initialize a chat session dictionary to keep context per user
user_chats = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    username = update.effective_user.username

    if ALLOWED_USER_ID:
        allowed = [u.strip() for u in ALLOWED_USER_ID.split(',')]
        if user_id not in allowed and username not in allowed:
            await update.message.reply_text("Sorry, you are not authorized to use this bot.")
            return

    # Reset chat history on /start
    user_chats[user_id] = model.start_chat(history=[])
    
    welcome_message = (
        "Hello! I am your personal AI assistant powered by Google Gemini.\n"
        "How can I help you today?"
    )
    await update.message.reply_text(welcome_message)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    username = update.effective_user.username

    if ALLOWED_USER_ID:
        allowed = [u.strip() for u in ALLOWED_USER_ID.split(',')]
        if user_id not in allowed and username not in allowed:
            # Ignore messages from unauthorized users silently or reply once.
            return

    user_message = update.message.text

    if not user_message:
        return

    # Get or create chat session
    if user_id not in user_chats:
        user_chats[user_id] = model.start_chat(history=[])
        
    chat = user_chats[user_id]

    try:
        # Send typing action
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        
        # Get response from Gemini
        response = chat.send_message(user_message)
        
        # Send response back to user
        await update.message.reply_text(response.text)
    except Exception as e:
        logger.error(f"Error communicating with Gemini: {e}")
        await update.message.reply_text("Sorry, I encountered an error while processing your request.")

def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
        logger.error("Missing TELEGRAM_BOT_TOKEN or GEMINI_API_KEY in environment variables.")
        return

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Fix for Python 3.10+ event loop RuntimeError on some environments
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    port = int(os.environ.get('PORT', 8080))

    if WEBHOOK_URL:
        logger.info(f"Starting webhook on port {port} with URL {WEBHOOK_URL}")
        # When using webhooks, the application starts its own HTTP server.
        # This completely replaces the need for the dummy server!
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            webhook_url=WEBHOOK_URL
        )
    else:
        logger.info("WEBHOOK_URL not found in environment. Falling back to long-polling.")
        app.run_polling()

if __name__ == "__main__":
    main()
