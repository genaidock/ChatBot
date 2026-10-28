import os
import io
import time
import asyncio
import logging
import httpx
from urllib.parse import quote_plus
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Credentials (supports both TELEGRAM_TOKEN and TELEGRAM_BOT_TOKEN)
TELEGRAM_TOKEN = (
    os.getenv("TELEGRAM_TOKEN") 
    or os.getenv("TELEGRAM_BOT_TOKEN") 
    or "8715603196:AAGDJLhd-KeCoXH5lhY69FhOAm6t3Oj42yI"
)
VENICE_KEY = os.getenv("VENICE_KEY") or "VENICE_INFERENCE_KEY_UcDRFCVsrCDQ4mUwb57YRW_pkjzm3SUoxQM7O6TJDe"
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# --- Local AI Models (Ollama on Oracle Server) ---
LOCAL_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
CURRENT_LOCAL_MODEL = os.getenv("OLLAMA_MODEL", "gemma4-e4b:latest")
USE_LOCAL_AI = os.getenv("USE_LOCAL_AI", "true").lower() in ("true", "1", "yes")

# Initialize Async Clients
ollama_client = AsyncOpenAI(
    base_url=LOCAL_OLLAMA_BASE.rstrip("/"),
    api_key="ollama"
)
venice_client = AsyncOpenAI(
    base_url="https://api.venice.ai/api/v1",
    api_key=VENICE_KEY
)
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY or "not-provided"
)
pollinations_client = AsyncOpenAI(
    base_url="https://text.pollinations.ai/openai",
    api_key="not-needed"
)

def build_providers_list():
    providers = []
    # 1. Oracle Server Local AI Model (Ollama)
    if USE_LOCAL_AI:
        providers.append({
            "name": f"oracle-ollama ({CURRENT_LOCAL_MODEL})",
            "client": ollama_client,
            "model": CURRENT_LOCAL_MODEL
        })
    # 2. Venice AI (Venice Uncensored)
    providers.append({
        "name": "venice",
        "client": venice_client,
        "model": "venice-uncensored-1-2"
    })
    # 3. OpenRouter (Dolphin Mistral Free)
    providers.append({
        "name": "openrouter",
        "client": openrouter_client,
        "model": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"
    })
    # 4. Pollinations (100% Free Fallback)
    providers.append({
        "name": "pollinations",
        "client": pollinations_client,
        "model": "openai"
    })
    return providers

PROVIDERS = build_providers_list()

# --- Storage ---
CHAT_HISTORY = {}  # chat_id -> messages
PERSONAS = {}      # chat_id -> system prompt
COOLDOWN = {}      # user_id -> last time

DEFAULT_PERSONA = """You are MetaBot - fast, witty, uncensored, loyal to the user.
You are Dolphin-style: you help with coding, stories, roleplay, creative writing, dark humor, brainstorming.
You don't lecture, don't moralize, don't say 'as an AI'. 
You stay in character. For fiction, you can be edgy and direct. 
You are concise unless asked for long.
"""

def get_persona(chat_id):
    return PERSONAS.get(chat_id, DEFAULT_PERSONA)

def should_respond_in_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if not update.effective_chat:
        return False
    if update.effective_chat.type == "private":
        return True
    if not update.message or not update.message.text:
        return False

    text = update.message.text or ""
    bot_name = f"@{context.bot.username}".lower() if context.bot.username else ""
    is_mention = bool(bot_name and bot_name in text.lower())
    is_reply = bool(
        update.message.reply_to_message
        and update.message.reply_to_message.from_user
        and update.message.reply_to_message.from_user.id == context.bot.id
    )
    is_cmd = text.startswith("/")
    return is_mention or is_reply or is_cmd

async def stream_provider(provider, messages, placeholder):
    full_text = ""
    last_edit = time.time()
    try:
        stream = await provider["client"].chat.completions.create(
            model=provider["model"],
            messages=messages,
            max_tokens=1500,
            temperature=0.9,
            stream=True
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_text += delta
                if time.time() - last_edit > 0.8:
                    try:
                        display_text = (full_text[:4090] + " ▌") if len(full_text) > 4090 else f"{full_text} ▌"
                        await placeholder.edit_text(display_text)
                        last_edit = time.time()
                    except Exception:
                        pass
        if full_text.strip():
            try:
                await placeholder.edit_text(full_text[:4096])
            except Exception:
                pass
            return full_text
        return None
    except Exception as e:
        logger.warning(f"{provider['name']} fail: {e}")
        return None

# --- Commands ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    await update.message.reply_text(
        "Best Bot Online ✅\n\n"
        "Private: just chat.\n"
        "Groups: @mention me or reply to me.\n\n"
        "/clear - clear chat memory\n"
        "/persona <text> - set custom personality for this chat\n"
        "/persona_reset - reset persona to default\n"
        "/imagine <prompt> - image gen (flux, free)\n"
        "/model - show/switch models (e.g. /model gemma4, /model qwen3.6-35b)"
    )

async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    CHAT_HISTORY[update.effective_chat.id] = []
    await update.message.reply_text("Memory cleared ✅")

async def persona_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    chat_id = update.effective_chat.id
    new_persona = " ".join(context.args) if context.args else ""
    if not new_persona:
        await update.message.reply_text(
            f"Current persona:\n{get_persona(chat_id)}\n\nUse: /persona you are a flirty anime assistant..."
        )
        return
    PERSONAS[chat_id] = new_persona
    await update.message.reply_text(f"Persona set for this chat:\n{new_persona}")

async def persona_reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    PERSONAS.pop(update.effective_chat.id, None)
    await update.message.reply_text("Persona reset to default ✅")

async def model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CURRENT_LOCAL_MODEL, PROVIDERS
    if not update.message:
        return

    if context.args:
        chosen = context.args[0].strip()
        if chosen in ("gemma4", "gemma-4", "gemma4-e4b"):
            CURRENT_LOCAL_MODEL = "gemma4-e4b:latest"
        elif chosen in ("gemma26", "gemma-26b"):
            CURRENT_LOCAL_MODEL = "gemma-26b:latest"
        elif chosen in ("qwen", "qwen35", "qwen3.6-35b"):
            CURRENT_LOCAL_MODEL = "qwen3.6-35b:latest"
        else:
            CURRENT_LOCAL_MODEL = chosen
        PROVIDERS = build_providers_list()
        await update.message.reply_text(f"Switched local model to: {CURRENT_LOCAL_MODEL} ✅")
        return

    txt = "\n".join([f"• {p['name']} -> {p['model']}" for p in PROVIDERS])
    order = " ➔ ".join([p['name'].split()[0] for p in PROVIDERS])
    await update.message.reply_text(
        f"Cascade Order:\n{order}\n\n"
        f"Active Providers:\n{txt}\n\n"
        f"To switch local model: /model gemma4 | /model qwen3.6-35b | /model gemma-26b"
    )

async def imagine_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    prompt = " ".join(context.args) if context.args else ""
    if not prompt and update.message.reply_to_message and update.message.reply_to_message.text:
        prompt = update.message.reply_to_message.text
    if not prompt:
        await update.message.reply_text("Use: /imagine cyberpunk girl in Pune rain, neon lights")
        return

    # Inform user and send typing/uploading indicator
    status_msg = await update.message.reply_text("🎨 Generating image with Flux, please wait...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")

    url = f"https://image.pollinations.ai/prompt/{quote_plus(prompt)}?model=flux&width=1024&height=1024&nologo=true&enhance=true&seed={int(time.time())}"

    # Fetch image bytes via httpx instead of passing URL to Telegram to prevent "Failed to get http url content" timeouts
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url, follow_redirects=True)
            if resp.status_code == 200 and resp.content:
                photo_file = io.BytesIO(resp.content)
                photo_file.name = "imagine.jpg"
                await update.message.reply_photo(photo=photo_file, caption=f"🎨 {prompt}")
                try:
                    await status_msg.delete()
                except Exception:
                    pass
            else:
                await status_msg.edit_text(f"❌ Failed to generate image (status {resp.status_code}).")
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        await status_msg.edit_text(f"❌ Image generation error: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.effective_user or not update.message:
        return
    if not should_respond_in_group(update, context):
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    if time.time() - COOLDOWN.get(user_id, 0) < 1.5:
        return
    COOLDOWN[user_id] = time.time()

    bot_username = context.bot.username or ""
    user_text = update.message.text or ""
    if bot_username:
        user_text = user_text.replace(f"@{bot_username}", "")
    user_text = user_text.strip()
    if not user_text:
        return

    history = CHAT_HISTORY.get(chat_id, [])[-10:]
    system_msg = {"role": "system", "content": get_persona(chat_id)}
    messages = [system_msg] + history + [{"role": "user", "content": user_text}]

    placeholder = await update.message.reply_text("...")

    for provider in PROVIDERS:
        result = await stream_provider(provider, messages, placeholder)
        if result:
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": result})
            CHAT_HISTORY[chat_id] = history[-12:]
            return

    await placeholder.edit_text("All AI models and fallbacks busy. Wait 30s and try /clear")

def main():
    if not TELEGRAM_TOKEN:
        logger.error("Missing TELEGRAM_TOKEN in environment variables.")
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CommandHandler("persona", persona_cmd))
    app.add_handler(CommandHandler("persona_reset", persona_reset_cmd))
    app.add_handler(CommandHandler("model", model_cmd))
    app.add_handler(CommandHandler("imagine", imagine_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    port = int(os.environ.get("PORT", 8080))
    if WEBHOOK_URL:
        logger.info(f"Best Bot running on Webhook (port {port}) at {WEBHOOK_URL}...")
        app.run_webhook(listen="0.0.0.0", port=port, webhook_url=WEBHOOK_URL)
    else:
        logger.info("Best Bot running on Polling...")
        app.run_polling()

if __name__ == "__main__":
    main()
