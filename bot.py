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
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# --- OmniRoute Gateway (Oracle Server / Port 20128) ---
OMNIROUTE_BASE = os.getenv("OMNIROUTE_BASE", "http://localhost:20128/v1")
OMNIROUTE_KEY = os.getenv("OMNIROUTE_KEY")
CURRENT_OMNIROUTE_MODEL = os.getenv("OMNIROUTE_MODEL", "auto")

# --- Local Ollama (Oracle Server / Port 11434) ---
LOCAL_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
CURRENT_LOCAL_MODEL = os.getenv("OLLAMA_MODEL", "gemma4-e4b:latest")
USE_LOCAL_AI = os.getenv("USE_LOCAL_AI", "true").lower() in ("true", "1", "yes")

# --- Cloud API Providers ---
VENICE_KEY = os.getenv("VENICE_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
POLLINATIONS_KEY = os.getenv("POLLINATIONS_KEY")

def build_providers_list():
    providers = []
    
    omniroute_provider = None
    if OMNIROUTE_BASE and OMNIROUTE_KEY:
        omniroute_client = AsyncOpenAI(
            base_url=OMNIROUTE_BASE.rstrip("/"),
            api_key=OMNIROUTE_KEY
        )
        omniroute_provider = {
            "name": f"omniroute ({CURRENT_OMNIROUTE_MODEL})",
            "client": omniroute_client,
            "model": CURRENT_OMNIROUTE_MODEL
        }

    ollama_provider = None
    if USE_LOCAL_AI:
        ollama_client = AsyncOpenAI(
            base_url=LOCAL_OLLAMA_BASE.rstrip("/"),
            api_key="ollama"
        )
        ollama_provider = {
            "name": f"oracle-ollama ({CURRENT_LOCAL_MODEL})",
            "client": ollama_client,
            "model": CURRENT_LOCAL_MODEL
        }

    # If the active local model is an uncensored model (like dolphin), prioritize local Ollama
    if "dolphin" in CURRENT_LOCAL_MODEL.lower():
        if ollama_provider:
            providers.append(ollama_provider)
        if omniroute_provider:
            providers.append(omniroute_provider)
    else:
        if omniroute_provider:
            providers.append(omniroute_provider)
        if ollama_provider:
            providers.append(ollama_provider)

    # 3. Venice AI (Venice Uncensored)
    if VENICE_KEY:
        venice_client = AsyncOpenAI(
            base_url="https://api.venice.ai/api/v1",
            api_key=VENICE_KEY
        )
        providers.append({
            "name": "venice",
            "client": venice_client,
            "model": "venice-uncensored-1-2"
        })

    # 4. OpenRouter
    if OPENROUTER_KEY:
        openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_KEY
        )
        providers.append({
            "name": "openrouter",
            "client": openrouter_client,
            "model": OPENROUTER_MODEL
        })

    # 5. Pollinations (Safety Net Text Completion)
    pollinations_client = AsyncOpenAI(
        base_url="https://text.pollinations.ai/openai",
        api_key=POLLINATIONS_KEY or "not-needed"
    )
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
Always communicate in fluent English unless the user explicitly speaks or asks in another language.
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
        stream = await asyncio.wait_for(
            provider["client"].chat.completions.create(
                model=provider["model"],
                messages=messages,
                max_tokens=1500,
                temperature=0.9,
                stream=True
            ),
            timeout=25.0
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
        "/imagine <prompt> - free AI image generation\n"
        "/model - show/switch models (e.g. /model dolmis, /model dolphin, /model qwen)"
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

async def dolmis_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CURRENT_OMNIROUTE_MODEL, CURRENT_LOCAL_MODEL, PROVIDERS
    if not update.message:
        return
    CURRENT_LOCAL_MODEL = "dolphin-mistral:7b"
    CURRENT_OMNIROUTE_MODEL = "auto/qwen"
    PROVIDERS = build_providers_list()
    await update.message.reply_text("Switched to Uncensored Dolphin Mistral 7B (`dolphin-mistral:7b`) 🐬⚡")

async def dolphin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CURRENT_OMNIROUTE_MODEL, CURRENT_LOCAL_MODEL, PROVIDERS
    if not update.message:
        return
    CURRENT_LOCAL_MODEL = "dolphin3:latest"
    CURRENT_OMNIROUTE_MODEL = "auto/qwen"
    PROVIDERS = build_providers_list()
    await update.message.reply_text("Switched to Uncensored Dolphin 3.0 (`dolphin3`) 🐬🔓")

async def model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CURRENT_OMNIROUTE_MODEL, CURRENT_LOCAL_MODEL, PROVIDERS
    if not update.message:
        return

    if context.args:
        chosen = context.args[0].strip().lower()
        if chosen in ("qwen", "qwen35", "qwen3.6-35b", "auto/qwen"):
            CURRENT_OMNIROUTE_MODEL = "auto/qwen"
            CURRENT_LOCAL_MODEL = "qwen3.6-35b"
            await update.message.reply_text("Switched model to Qwen (`auto/qwen`) ✅")
        elif chosen in ("gemma", "gemma4", "gemma-4", "gemma4-e4b"):
            CURRENT_OMNIROUTE_MODEL = "auto/gemma"
            CURRENT_LOCAL_MODEL = "gemma4-e4b:latest"
            await update.message.reply_text("Switched model to Gemma (`auto/gemma` / `gemma4-e4b`) ✅")
        elif chosen in ("gemma26", "gemma-26b"):
            CURRENT_LOCAL_MODEL = "gemma-26b:latest"
            await update.message.reply_text(f"Switched local Ollama model to: {CURRENT_LOCAL_MODEL} ✅")
        elif chosen in ("dolphin", "dolphin3", "uncensored"):
            CURRENT_LOCAL_MODEL = "dolphin3:latest"
            CURRENT_OMNIROUTE_MODEL = "auto/qwen"
            await update.message.reply_text(
                "Switched to Uncensored Dolphin 3.0 (`dolphin3`) 🐬🔓"
            )
        elif chosen in ("dolmis", "dol-mis", "dolphin-mistral", "mistral-dolphin", "dolphin-mistral:7b"):
            CURRENT_LOCAL_MODEL = "dolphin-mistral:7b"
            CURRENT_OMNIROUTE_MODEL = "auto/qwen"
            await update.message.reply_text(
                "Switched to Uncensored Dolphin Mistral 7B (`dolphin-mistral:7b`) 🐬⚡"
            )
        elif chosen in ("deepseek", "auto/deepseek"):
            CURRENT_OMNIROUTE_MODEL = "auto/deepseek"
            await update.message.reply_text("Switched model to DeepSeek (`auto/deepseek`) ✅")
        elif chosen in ("coding", "code", "best-coding", "auto/best-coding"):
            CURRENT_OMNIROUTE_MODEL = "auto/best-coding"
            await update.message.reply_text("Switched model to Best Coding (`auto/best-coding`) ✅")
        elif chosen in ("chat", "best-chat", "auto/best-chat"):
            CURRENT_OMNIROUTE_MODEL = "auto/best-chat"
            await update.message.reply_text("Switched model to Best Chat (`auto/best-chat`) ✅")
        elif chosen in ("auto", "default"):
            CURRENT_OMNIROUTE_MODEL = "auto"
            await update.message.reply_text("Switched model to Auto-Routing (`auto`) ✅")
        else:
            CURRENT_OMNIROUTE_MODEL = chosen
            await update.message.reply_text(f"Switched OmniRoute model to: `{CURRENT_OMNIROUTE_MODEL}` ✅")

        PROVIDERS = build_providers_list()
        return

    txt = "\n".join([f"• {p['name']} -> {p['model']}" for p in PROVIDERS])
    order = " ➔ ".join([p['name'].split()[0] for p in PROVIDERS])
    await update.message.reply_text(
        f"Cascade Order:\n{order}\n\n"
        f"Active Providers:\n{txt}\n\n"
        f"Switch models on the fly:\n"
        f"• `/model dolmis` - Uncensored Dolphin Mistral (7B, Fast)\n"
        f"• `/model dolphin` - Uncensored Dolphin 3.0 (8B)\n"
        f"• `/model qwen` - High-speed Qwen AI\n"
        f"• `/model gemma` - Gemma 4\n"
        f"• `/model deepseek` - DeepSeek\n"
        f"• `/model auto` - Smart Auto-Routing\n"
        f"• `/model auto/best-coding` - Best Coding Model"
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

    status_msg = await update.message.reply_text("🎨 Generating image, please wait...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "image/*,*/*;q=0.8"
    }
    if POLLINATIONS_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_KEY}"

    # Build URLs: with API key, full Flux and nologo=true are unlocked!
    encoded_prompt = quote_plus(prompt)
    if POLLINATIONS_KEY:
        urls_to_try = [
            f"https://image.pollinations.ai/prompt/{encoded_prompt}?model=flux&width=1024&height=1024&nologo=true",
            f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024"
        ]
    else:
        urls_to_try = [
            f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024",
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        ]

    image_bytes = None
    last_err = ""

    for url in urls_to_try:
        try:
            async with httpx.AsyncClient(timeout=45.0, headers=headers) as client:
                resp = await client.get(url, follow_redirects=True)
                content_type = resp.headers.get("content-type", "")
                if resp.status_code == 200 and resp.content and ("image" in content_type or resp.content[:2] == b"\xff\xd8"):
                    image_bytes = resp.content
                    break
                else:
                    last_err = f"Status {resp.status_code}"
        except Exception as e:
            last_err = str(e)
            logger.warning(f"Pollinations fetch failed: {e}")

    if image_bytes:
        try:
            caption_text = f"🎨 {prompt[:1000]}"
            await update.message.reply_photo(
                photo=image_bytes,
                filename="imagine.jpg",
                caption=caption_text,
                write_timeout=60.0
            )
            try:
                await status_msg.delete()
            except Exception:
                pass
            return
        except Exception as e:
            logger.error(f"Failed to send photo to Telegram: {e}")
            await status_msg.edit_text(f"❌ Telegram upload error: {e}")
    else:
        await status_msg.edit_text(f"❌ Failed to generate image ({last_err or 'Service busy'}). Please try again.")

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

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .connect_timeout(30.0)
        .read_timeout(60.0)
        .write_timeout(60.0)
        .build()
    )
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CommandHandler("persona", persona_cmd))
    app.add_handler(CommandHandler("persona_reset", persona_reset_cmd))
    app.add_handler(CommandHandler("model", model_cmd))
    app.add_handler(CommandHandler("dolmis", dolmis_cmd))
    app.add_handler(CommandHandler("dolphin", dolphin_cmd))
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
