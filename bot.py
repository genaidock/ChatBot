import os
import io
import time
import asyncio
import logging
import html
import httpx
from urllib.parse import quote_plus
from dotenv import load_dotenv

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BotCommand,
    constants
)
from telegram.error import BadRequest, RetryAfter, TelegramError
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    PicklePersistence,
    AIORateLimiter,
    filters,
    ContextTypes
)
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Credentials
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# --- OmniRoute Gateway (Oracle Server / Port 20128) ---
OMNIROUTE_BASE = os.getenv("OMNIROUTE_BASE", "http://localhost:20128/v1")
OMNIROUTE_KEY = os.getenv("OMNIROUTE_KEY")
DEFAULT_OMNIROUTE_MODEL = os.getenv("OMNIROUTE_MODEL", "auto")

# --- Local Ollama (Oracle Server / Port 11434) ---
LOCAL_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_LOCAL_MODEL = os.getenv("OLLAMA_MODEL", "gemma4-e4b:latest")
USE_LOCAL_AI = os.getenv("USE_LOCAL_AI", "true").lower() in ("true", "1", "yes")

# --- Cloud API Providers ---
VENICE_KEY = os.getenv("VENICE_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
POLLINATIONS_KEY = os.getenv("POLLINATIONS_KEY")

# --- Single Persistent Clients (reused across requests) ---
OMNIROUTE_CLIENT = (
    AsyncOpenAI(base_url=OMNIROUTE_BASE.rstrip("/"), api_key=OMNIROUTE_KEY)
    if OMNIROUTE_BASE and OMNIROUTE_KEY
    else None
)

OLLAMA_CLIENT = (
    AsyncOpenAI(base_url=LOCAL_OLLAMA_BASE.rstrip("/"), api_key="ollama")
    if USE_LOCAL_AI and LOCAL_OLLAMA_BASE
    else None
)

VENICE_CLIENT = (
    AsyncOpenAI(base_url="https://api.venice.ai/api/v1", api_key=VENICE_KEY)
    if VENICE_KEY
    else None
)

OPENROUTER_CLIENT = (
    AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
    if OPENROUTER_KEY
    else None
)

POLLINATIONS_CLIENT = AsyncOpenAI(
    base_url="https://text.pollinations.ai/openai",
    api_key=POLLINATIONS_KEY or "not-needed"
)

# --- Default Persona ---
DEFAULT_PERSONA = """You are MetaBot - fast, witty, uncensored, loyal to the user.
You are Dolphin-style: you help with coding, stories, roleplay, creative writing, dark humor, brainstorming.
You don't lecture, don't moralize, don't say 'as an AI'. 
You stay in character. For fiction, you can be edgy and direct. 
You are concise unless asked for long.
Always communicate in fluent English unless the user explicitly speaks or asks in another language.
"""

# In-memory storage fallbacks (used when Context is unavailable or transient)
_CHAT_HISTORY = {}
_PERSONAS = {}
_CHAT_MODELS = {}
COOLDOWN = {}

# --- Persistent Context Helpers (PicklePersistence integration) ---
def get_chat_config(chat_id: int, context: ContextTypes.DEFAULT_TYPE = None) -> dict:
    default_priority = "local" if "dolphin" in DEFAULT_LOCAL_MODEL.lower() else "omniroute"
    default_cfg = {
        "local_model": DEFAULT_LOCAL_MODEL,
        "omniroute_model": DEFAULT_OMNIROUTE_MODEL,
        "priority": default_priority
    }
    if context and getattr(context, "chat_data", None) is not None:
        if "model_config" not in context.chat_data:
            context.chat_data["model_config"] = default_cfg.copy()
        return context.chat_data["model_config"]
    if chat_id not in _CHAT_MODELS:
        _CHAT_MODELS[chat_id] = default_cfg.copy()
    return _CHAT_MODELS[chat_id]

def set_chat_config(chat_id: int, cfg: dict, context: ContextTypes.DEFAULT_TYPE = None):
    if context and getattr(context, "chat_data", None) is not None:
        context.chat_data["model_config"] = cfg
    _CHAT_MODELS[chat_id] = cfg

def get_persona(chat_id: int, context: ContextTypes.DEFAULT_TYPE = None) -> str:
    if context and getattr(context, "chat_data", None) is not None:
        return context.chat_data.get("persona", DEFAULT_PERSONA)
    return _PERSONAS.get(chat_id, DEFAULT_PERSONA)

def set_persona(chat_id: int, persona: str, context: ContextTypes.DEFAULT_TYPE = None):
    if context and getattr(context, "chat_data", None) is not None:
        context.chat_data["persona"] = persona
    _PERSONAS[chat_id] = persona

def reset_persona(chat_id: int, context: ContextTypes.DEFAULT_TYPE = None):
    if context and getattr(context, "chat_data", None) is not None:
        context.chat_data.pop("persona", None)
    _PERSONAS.pop(chat_id, None)

def get_chat_history(chat_id: int, context: ContextTypes.DEFAULT_TYPE = None) -> list:
    if context and getattr(context, "chat_data", None) is not None:
        if "history" not in context.chat_data:
            context.chat_data["history"] = []
        return context.chat_data["history"]
    if chat_id not in _CHAT_HISTORY:
        _CHAT_HISTORY[chat_id] = []
    return _CHAT_HISTORY[chat_id]

def set_chat_history(chat_id: int, history: list, context: ContextTypes.DEFAULT_TYPE = None):
    trimmed = history[-12:]
    if context and getattr(context, "chat_data", None) is not None:
        context.chat_data["history"] = trimmed
    _CHAT_HISTORY[chat_id] = trimmed

def clear_chat_history(chat_id: int, context: ContextTypes.DEFAULT_TYPE = None):
    if context and getattr(context, "chat_data", None) is not None:
        context.chat_data["history"] = []
    _CHAT_HISTORY[chat_id] = []

# --- Provider Cascade Builder ---
def build_providers_for_chat(chat_id: int, context: ContextTypes.DEFAULT_TYPE = None):
    cfg = get_chat_config(chat_id, context)
    local_m = cfg["local_model"]
    omni_m = cfg["omniroute_model"]
    priority = cfg["priority"]

    providers = []

    omniroute_provider = None
    if OMNIROUTE_CLIENT:
        omniroute_provider = {
            "name": f"omniroute ({omni_m})",
            "client": OMNIROUTE_CLIENT,
            "model": omni_m
        }

    ollama_provider = None
    if USE_LOCAL_AI and OLLAMA_CLIENT:
        ollama_provider = {
            "name": f"oracle-ollama ({local_m})",
            "client": OLLAMA_CLIENT,
            "model": local_m
        }

    # Order based on chat priority
    if priority == "local":
        if ollama_provider:
            providers.append(ollama_provider)
        if omniroute_provider:
            providers.append(omniroute_provider)
    else:
        if omniroute_provider:
            providers.append(omniroute_provider)
        if ollama_provider:
            providers.append(ollama_provider)

    # Venice AI (Uncensored Cloud)
    if VENICE_CLIENT:
        providers.append({
            "name": "venice",
            "client": VENICE_CLIENT,
            "model": "venice-uncensored-1-2"
        })

    # OpenRouter
    if OPENROUTER_CLIENT:
        providers.append({
            "name": "openrouter",
            "client": OPENROUTER_CLIENT,
            "model": OPENROUTER_MODEL
        })

    # Pollinations (Safety Net Text Completion)
    providers.append({
        "name": "pollinations",
        "client": POLLINATIONS_CLIENT,
        "model": "openai"
    })

    return providers

async def check_ollama_models():
    """Live health check against Ollama /api/tags"""
    try:
        base = LOCAL_OLLAMA_BASE.rstrip("/").replace("/v1", "")
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(f"{base}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name") for m in data.get("models", [])]
                return True, models, ""
            return False, [], f"HTTP {resp.status_code}"
    except Exception as e:
        return False, [], str(e)

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

# --- Safe Message Rendering & Edit Helper ---
async def safe_edit_message(target, text: str, reply_markup=None, parse_mode=None):
    """
    Safely edits a Telegram message or callback query.
    Catches BadRequest parsing errors and falls back to plain text.
    Suppresses 'Message is not modified'.
    """
    text_to_send = text[:4096] if len(text) > 4096 else text
    try:
        if hasattr(target, "edit_text"):
            return await target.edit_text(
                text=text_to_send,
                reply_markup=reply_markup,
                parse_mode=parse_mode
            )
        elif hasattr(target, "edit_message_text"):
            return await target.edit_message_text(
                text=text_to_send,
                reply_markup=reply_markup,
                parse_mode=parse_mode
            )
    except BadRequest as e:
        err = str(e).lower()
        if "message is not modified" in err:
            return None
        if "can't parse entities" in err or "entity" in err or "parse" in err:
            logger.debug(f"Markdown/HTML parse failed; falling back to raw text. Error: {e}")
            try:
                if hasattr(target, "edit_text"):
                    return await target.edit_text(
                        text=text_to_send,
                        reply_markup=reply_markup,
                        parse_mode=None
                    )
                elif hasattr(target, "edit_message_text"):
                    return await target.edit_message_text(
                        text=text_to_send,
                        reply_markup=reply_markup,
                        parse_mode=None
                    )
            except Exception as e2:
                logger.debug(f"Fallback plain-text edit failed: {e2}")
        else:
            logger.debug(f"BadRequest editing message: {e}")
    except Exception as e:
        logger.debug(f"safe_edit_message failed: {e}")
    return None

# --- Continuous Typing Indicator Worker ---
async def typing_worker(chat_id: int, bot, stop_event: asyncio.Event):
    """Periodically triggers Telegram typing action until generation starts or completes."""
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4.5)
        except asyncio.TimeoutError:
            continue

# --- Streaming Provider Engine ---
async def stream_provider(provider, messages, placeholder, stop_typing_event: asyncio.Event = None):
    full_text = ""
    last_edit = time.time()
    is_ollama = "ollama" in provider["name"].lower()
    # On CPU servers, loading a 4-5GB model into RAM can take 20-40 seconds
    timeout_val = 85.0 if is_ollama else 35.0

    extra_kwargs = {}
    if is_ollama:
        # Keep model resident in RAM for 24h
        extra_kwargs["extra_body"] = {"keep_alive": "24h"}

    try:
        stream = await asyncio.wait_for(
            provider["client"].chat.completions.create(
                model=provider["model"],
                messages=messages,
                max_tokens=1500,
                temperature=0.9,
                stream=True,
                **extra_kwargs
            ),
            timeout=timeout_val
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                if stop_typing_event and not stop_typing_event.is_set():
                    stop_typing_event.set()
                full_text += delta
                if time.time() - last_edit > 0.8:
                    display_text = (full_text[:4090] + " ▌") if len(full_text) > 4090 else f"{full_text} ▌"
                    await safe_edit_message(placeholder, display_text)
                    last_edit = time.time()

        if full_text.strip():
            await safe_edit_message(placeholder, full_text)
            return full_text, None
        return None, "empty response"
    except Exception as e:
        err_msg = str(e)
        logger.warning(f"{provider['name']} fail: {err_msg}")
        return None, err_msg

# --- UI Keyboard Builders ---
def build_start_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🎯 Switch AI Model", callback_data="act:open_model"),
            InlineKeyboardButton("🔍 Check Ollama", callback_data="act:ollama"),
        ],
        [
            InlineKeyboardButton("🎨 Imagine Image", switch_inline_query_current_chat="/imagine "),
            InlineKeyboardButton("🧹 Clear Memory", callback_data="act:clear"),
        ]
    ])

def build_model_keyboard(cfg: dict):
    local_m = cfg.get("local_model", DEFAULT_LOCAL_MODEL)
    omni_m = cfg.get("omniroute_model", DEFAULT_OMNIROUTE_MODEL)
    priority = cfg.get("priority", "omniroute")

    def mark(cond: bool) -> str:
        return "✅ " if cond else ""

    keyboard = [
        [
            InlineKeyboardButton(
                f"{mark(local_m == 'dolphin3:latest' and priority == 'local')}🐬 Dolphin 3.0",
                callback_data="model:dolphin"
            ),
            InlineKeyboardButton(
                f"{mark(local_m == 'dolphin-mistral:7b' and priority == 'local')}⚡ Dol-Mistral",
                callback_data="model:dolmis"
            ),
        ],
        [
            InlineKeyboardButton(
                f"{mark(omni_m == 'auto/deepseek')}🧠 DeepSeek",
                callback_data="model:deepseek"
            ),
            InlineKeyboardButton(
                f"{mark(omni_m == 'auto/gemma')}💎 Gemma 4",
                callback_data="model:gemma"
            ),
        ],
        [
            InlineKeyboardButton(
                f"{mark(omni_m == 'auto/best-coding')}💻 Coding AI",
                callback_data="model:coding"
            ),
            InlineKeyboardButton(
                f"{mark(omni_m == 'auto/best-chat')}💬 Best Chat",
                callback_data="model:chat"
            ),
        ],
        [
            InlineKeyboardButton(
                f"{mark(omni_m == 'auto' and priority == 'omniroute')}🎯 Auto-Routing",
                callback_data="model:auto"
            ),
            InlineKeyboardButton(
                f"🔄 Priority: {priority.upper()}",
                callback_data="model:toggle_priority"
            ),
        ],
        [
            InlineKeyboardButton("🔍 Check Ollama", callback_data="act:ollama"),
            InlineKeyboardButton("🧹 Clear Memory", callback_data="act:clear"),
            InlineKeyboardButton("🔄 Refresh", callback_data="act:refresh_model"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def format_model_dashboard_text(chat_id: int, context: ContextTypes.DEFAULT_TYPE = None) -> str:
    cfg = get_chat_config(chat_id, context)
    providers = build_providers_for_chat(chat_id, context)
    txt = "\n".join([f"• {p['name']} -> <code>{p['model']}</code>" for p in providers])
    order = " ➔ ".join([p['name'].split()[0] for p in providers])

    return (
        f"🎯 <b>Active Configuration for this Chat:</b>\n"
        f"• Local Ollama: <code>{cfg['local_model']}</code>\n"
        f"• OmniRoute: <code>{cfg['omniroute_model']}</code>\n"
        f"• Priority: <b>{cfg['priority'].upper()}</b>\n\n"
        f"<b>Cascade Failover Order:</b>\n{order}\n\n"
        f"<b>Active Providers:</b>\n{txt}\n\n"
        f"<i>Tap any button below to switch models or toggle priority instantly:</i>"
    )

# --- Commands ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    msg = (
        "🤖 <b>Welcome to Best Bot Online</b>\n\n"
        "💬 <b>How to Chat:</b>\n"
        "• Private: Just type and send your message.\n"
        "• Groups: @mention me or reply to my message.\n\n"
        "🧠 <b>Available AI Models:</b>\n\n"
        "🐬 <i>Uncensored Models (Zero Guardrails / Refusals):</i>\n"
        "• <b>Dolphin Mistral 7B</b> (/dolmis) — Fast, direct, uncensored\n"
        "• <b>Dolphin 3.0 Llama 8B</b> (/dolphin) — Deep logic, uncensored\n\n"
        "⚡ <i>Cloud AI Models (High Performance):</i>\n"
        "• <b>DeepSeek</b> (/deepseek) — Math, reasoning & analysis\n"
        "• <b>Google Gemma 4</b> (/gemma) — Google latest architecture\n"
        "• <b>Best Coding AI</b> (/coding) — Programming & debugging\n"
        "• <b>Best Chat AI</b> (/chat) — Fluid everyday conversation\n"
        "• <b>Smart Auto-Routing</b> (/auto) — Auto-selects optimal model\n\n"
        "⚙️ <b>Control Commands:</b>\n"
        "• /model — Interactive dashboard to switch models & view cascade\n"
        "• /ollama — Check Local Ollama models & connection status\n"
        "• /clear — Clear chat memory history\n"
        "• /imagine &lt;prompt&gt; — Free Flux AI image generation\n"
        "• /persona &lt;text&gt; — Set custom personality\n"
        "• /persona_reset — Reset persona to default"
    )
    await update.message.reply_text(
        msg,
        parse_mode="HTML",
        reply_markup=build_start_keyboard()
    )

async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    clear_chat_history(update.effective_chat.id, context)
    await update.message.reply_text("Memory cleared ✅")

async def persona_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    chat_id = update.effective_chat.id
    new_persona = " ".join(context.args) if context.args else ""
    if not new_persona:
        current = get_persona(chat_id, context)
        await update.message.reply_text(
            f"Current persona:\n{current}\n\nUse: /persona you are a flirty anime assistant..."
        )
        return
    set_persona(chat_id, new_persona, context)
    await update.message.reply_text(f"Persona set for this chat:\n{new_persona}")

async def persona_reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    reset_persona(update.effective_chat.id, context)
    await update.message.reply_text("Persona reset to default ✅")

async def dolmis_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    chat_id = update.effective_chat.id
    set_chat_config(chat_id, {
        "local_model": "dolphin-mistral:7b",
        "omniroute_model": "auto",
        "priority": "local"
    }, context)
    await update.message.reply_text(
        "🐬⚡ Switched this chat to <b>Uncensored Dolphin Mistral 7B</b> (<code>dolphin-mistral:7b</code>)\n"
        "Priority: <b>Local Ollama</b>.\n"
        "Tip: Use /ollama to check if this model is downloaded on the server.",
        parse_mode="HTML"
    )

async def dolphin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    chat_id = update.effective_chat.id
    set_chat_config(chat_id, {
        "local_model": "dolphin3:latest",
        "omniroute_model": "auto",
        "priority": "local"
    }, context)
    await update.message.reply_text(
        "🐬🔓 Switched this chat to <b>Uncensored Dolphin 3.0</b> (<code>dolphin3</code>)\n"
        "Priority: <b>Local Ollama</b>.\n"
        "Tip: Use /ollama to check if this model is downloaded on the server.",
        parse_mode="HTML"
    )

async def deepseek_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    chat_id = update.effective_chat.id
    cfg = get_chat_config(chat_id, context)
    cfg["omniroute_model"] = "auto/deepseek"
    cfg["priority"] = "omniroute"
    set_chat_config(chat_id, cfg, context)
    await update.message.reply_text("Switched to DeepSeek (`auto/deepseek`) 🧠✅")

async def gemma_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    chat_id = update.effective_chat.id
    set_chat_config(chat_id, {
        "local_model": "gemma4-e4b:latest",
        "omniroute_model": "auto/gemma",
        "priority": "omniroute"
    }, context)
    await update.message.reply_text("Switched to Google Gemma 4 (`auto/gemma` / `gemma4-e4b`) 💎✅")

async def coding_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    chat_id = update.effective_chat.id
    cfg = get_chat_config(chat_id, context)
    cfg["omniroute_model"] = "auto/best-coding"
    cfg["priority"] = "omniroute"
    set_chat_config(chat_id, cfg, context)
    await update.message.reply_text("Switched to Best Coding AI (`auto/best-coding`) 💻✅")

async def chat_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    chat_id = update.effective_chat.id
    cfg = get_chat_config(chat_id, context)
    cfg["omniroute_model"] = "auto/best-chat"
    cfg["priority"] = "omniroute"
    set_chat_config(chat_id, cfg, context)
    await update.message.reply_text("Switched to Best Chat AI (`auto/best-chat`) 💬✅")

async def auto_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    chat_id = update.effective_chat.id
    cfg = get_chat_config(chat_id, context)
    cfg["omniroute_model"] = "auto"
    cfg["priority"] = "omniroute"
    set_chat_config(chat_id, cfg, context)
    await update.message.reply_text("Switched to Smart Auto-Routing (`auto`) 🎯✅")

async def ollama_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return
    status_msg = await update.message.reply_text("🔍 Inspecting Local Ollama service...")
    ok, models, err = await check_ollama_models()
    if ok:
        models_txt = "\n".join([f"• <code>{m}</code>" for m in models]) if models else "<i>(No models downloaded)</i>"
        has_dolphin = any("dolphin" in m.lower() for m in models)
        warning_box = ""
        if not has_dolphin:
            warning_box = (
                "\n\n⚠️ <b>Dolphin models are missing!</b>\n"
                "To use /dolphin or /dolmis, pull them on your Oracle terminal:\n"
                "<code>ollama pull dolphin3</code>\n"
                "<code>ollama pull dolphin-mistral:7b</code>"
            )
        await safe_edit_message(
            status_msg,
            f"🟢 <b>Local Ollama Status: ONLINE</b>\n"
            f"Endpoint: <code>{LOCAL_OLLAMA_BASE}</code>\n\n"
            f"<b>Available Local Models:</b>\n{models_txt}{warning_box}",
            parse_mode="HTML"
        )
    else:
        await safe_edit_message(
            status_msg,
            f"🔴 <b>Local Ollama Status: OFFLINE / UNREACHABLE</b>\n\n"
            f"Endpoint: <code>{LOCAL_OLLAMA_BASE}</code>\n"
            f"Error: <code>{err}</code>\n\n"
            f"Ensure Ollama is running on the Oracle server with: <code>systemctl status ollama</code>",
            parse_mode="HTML"
        )

async def model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return

    chat_id = update.effective_chat.id

    if context.args:
        chosen = context.args[0].strip().lower()
        if chosen in ("check", "ollama", "status"):
            await ollama_cmd(update, context)
            return
        elif chosen in ("gemma", "gemma4", "gemma-4", "gemma4-e4b"):
            set_chat_config(chat_id, {
                "local_model": "gemma4-e4b:latest",
                "omniroute_model": "auto/gemma",
                "priority": "omniroute"
            }, context)
            await update.message.reply_text("Switched model to Gemma (`auto/gemma` / `gemma4-e4b`) ✅")
        elif chosen in ("gemma26", "gemma-26b"):
            cfg = get_chat_config(chat_id, context)
            cfg["local_model"] = "gemma-26b:latest"
            cfg["priority"] = "local"
            set_chat_config(chat_id, cfg, context)
            await update.message.reply_text("Switched local Ollama model to: `gemma-26b:latest` ✅")
        elif chosen in ("dolphin", "dolphin3", "uncensored"):
            set_chat_config(chat_id, {
                "local_model": "dolphin3:latest",
                "omniroute_model": "auto",
                "priority": "local"
            }, context)
            await update.message.reply_text("Switched to Uncensored Dolphin 3.0 (`dolphin3`) 🐬🔓")
        elif chosen in ("dolmis", "dol-mis", "dolphin-mistral", "mistral-dolphin", "dolphin-mistral:7b"):
            set_chat_config(chat_id, {
                "local_model": "dolphin-mistral:7b",
                "omniroute_model": "auto",
                "priority": "local"
            }, context)
            await update.message.reply_text("Switched to Uncensored Dolphin Mistral 7B (`dolphin-mistral:7b`) 🐬⚡")
        elif chosen in ("deepseek", "auto/deepseek"):
            cfg = get_chat_config(chat_id, context)
            cfg["omniroute_model"] = "auto/deepseek"
            cfg["priority"] = "omniroute"
            set_chat_config(chat_id, cfg, context)
            await update.message.reply_text("Switched model to DeepSeek (`auto/deepseek`) ✅")
        elif chosen in ("coding", "code", "best-coding", "auto/best-coding"):
            cfg = get_chat_config(chat_id, context)
            cfg["omniroute_model"] = "auto/best-coding"
            cfg["priority"] = "omniroute"
            set_chat_config(chat_id, cfg, context)
            await update.message.reply_text("Switched model to Best Coding (`auto/best-coding`) ✅")
        elif chosen in ("chat", "best-chat", "auto/best-chat"):
            cfg = get_chat_config(chat_id, context)
            cfg["omniroute_model"] = "auto/best-chat"
            cfg["priority"] = "omniroute"
            set_chat_config(chat_id, cfg, context)
            await update.message.reply_text("Switched model to Best Chat (`auto/best-chat`) ✅")
        elif chosen in ("auto", "default"):
            cfg = get_chat_config(chat_id, context)
            cfg["omniroute_model"] = "auto"
            cfg["priority"] = "omniroute"
            set_chat_config(chat_id, cfg, context)
            await update.message.reply_text("Switched model to Auto-Routing (`auto`) ✅")
        else:
            cfg = get_chat_config(chat_id, context)
            cfg["omniroute_model"] = chosen
            set_chat_config(chat_id, cfg, context)
            await update.message.reply_text(f"Switched OmniRoute model to: `{chosen}` ✅")
        return

    # No arguments provided: show interactive dashboard with InlineKeyboardMarkup
    cfg = get_chat_config(chat_id, context)
    keyboard = build_model_keyboard(cfg)
    dashboard_text = format_model_dashboard_text(chat_id, context)

    await update.message.reply_text(
        dashboard_text,
        parse_mode="HTML",
        reply_markup=keyboard
    )

# --- Interactive Callback Query Handler ---
async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query or not query.message or not update.effective_chat:
        return

    chat_id = update.effective_chat.id
    data = query.data or ""
    cfg = get_chat_config(chat_id, context)

    if data.startswith("model:"):
        choice = data.split(":", 1)[1]
        toast = ""
        if choice == "dolphin":
            cfg["local_model"] = "dolphin3:latest"
            cfg["priority"] = "local"
            toast = "Switched to Dolphin 3.0 (Local Ollama) 🐬🔓"
        elif choice == "dolmis":
            cfg["local_model"] = "dolphin-mistral:7b"
            cfg["priority"] = "local"
            toast = "Switched to Dolphin Mistral 7B (Local Ollama) 🐬⚡"
        elif choice == "deepseek":
            cfg["omniroute_model"] = "auto/deepseek"
            cfg["priority"] = "omniroute"
            toast = "Switched to DeepSeek 🧠"
        elif choice == "gemma":
            cfg["local_model"] = "gemma4-e4b:latest"
            cfg["omniroute_model"] = "auto/gemma"
            cfg["priority"] = "omniroute"
            toast = "Switched to Google Gemma 4 💎"
        elif choice == "coding":
            cfg["omniroute_model"] = "auto/best-coding"
            cfg["priority"] = "omniroute"
            toast = "Switched to Best Coding AI 💻"
        elif choice == "chat":
            cfg["omniroute_model"] = "auto/best-chat"
            cfg["priority"] = "omniroute"
            toast = "Switched to Best Chat AI 💬"
        elif choice == "auto":
            cfg["omniroute_model"] = "auto"
            cfg["priority"] = "omniroute"
            toast = "Switched to Smart Auto-Routing 🎯"
        elif choice == "toggle_priority":
            cfg["priority"] = "omniroute" if cfg.get("priority") == "local" else "local"
            toast = f"Priority toggled to: {cfg['priority'].upper()} 🔄"

        set_chat_config(chat_id, cfg, context)
        await query.answer(toast)

        new_text = format_model_dashboard_text(chat_id, context)
        new_keyboard = build_model_keyboard(cfg)
        await safe_edit_message(query.message, new_text, reply_markup=new_keyboard, parse_mode="HTML")

    elif data == "act:open_model" or data == "act:refresh_model":
        await query.answer("Dashboard updated 🔄")
        new_text = format_model_dashboard_text(chat_id, context)
        new_keyboard = build_model_keyboard(cfg)
        await safe_edit_message(query.message, new_text, reply_markup=new_keyboard, parse_mode="HTML")

    elif data == "act:clear":
        clear_chat_history(chat_id, context)
        await query.answer("Chat memory cleared! ✅")
        try:
            await query.message.reply_text("Memory cleared ✅")
        except Exception:
            pass

    elif data == "act:ollama":
        await query.answer("Checking Ollama status...")
        ok, models, err = await check_ollama_models()
        if ok:
            models_txt = ", ".join(models[:4]) if models else "None"
            status_text = f"Ollama is ONLINE 🟢 ({len(models)} models: {models_txt})"
        else:
            status_text = f"Ollama is OFFLINE 🔴 ({err[:40]})"
        await query.answer(status_text, show_alert=True)

# --- Flux AI Image Generation ---
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
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.UPLOAD_PHOTO)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "image/*,*/*;q=0.8"
    }
    if POLLINATIONS_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_KEY}"

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
            await safe_edit_message(status_msg, f"❌ Telegram upload error: {e}")
    else:
        await safe_edit_message(status_msg, f"❌ Failed to generate image ({last_err or 'Service busy'}). Please try again.")

# --- Chat Message Handler ---
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

    history = get_chat_history(chat_id, context)[-10:]
    system_msg = {"role": "system", "content": get_persona(chat_id, context)}
    messages = [system_msg] + history + [{"role": "user", "content": user_text}]

    placeholder = await update.message.reply_text("...")

    providers = build_providers_for_chat(chat_id, context)
    if not providers:
        await safe_edit_message(placeholder, "No AI providers configured.")
        return

    preferred_provider = providers[0]
    preferred_name = preferred_provider["name"]
    first_error = None

    # Continuous typing indicator while waiting for the first token
    stop_typing_event = asyncio.Event()
    typing_task = asyncio.create_task(
        typing_worker(chat_id, context.bot, stop_typing_event)
    )

    try:
        for i, provider in enumerate(providers):
            result, err = await stream_provider(provider, messages, placeholder, stop_typing_event)
            if result:
                stop_typing_event.set()
                # If preferred provider failed and cascade fell back:
                if i > 0:
                    try:
                        reason = f" ({first_error[:100]})" if first_error else ""
                        hint = ""
                        err_lower = (first_error or "").lower()
                        if "oracle-ollama" in preferred_name and ("not found" in err_lower or "404" in err_lower):
                            hint = f"\n💡 _Model not downloaded yet. Run on server:_ `ollama pull {preferred_provider['model']}`"
                        elif "oracle-ollama" in preferred_name and ("connection refused" in err_lower or "connect" in err_lower):
                            hint = f"\n💡 _Ollama unreachable at {LOCAL_OLLAMA_BASE}_"

                        failover_note = (
                            f"\n\n⚠️ _Note: Preferred {preferred_name} failed{reason}. "
                            f"Responded via {provider['name']} instead._{hint}"
                        )
                        final_text = (result[:3800] + failover_note) if len(result) > 3800 else (result + failover_note)
                        await safe_edit_message(placeholder, final_text, parse_mode="Markdown")
                    except Exception:
                        pass

                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": result})
                set_chat_history(chat_id, history, context)
                return
            elif i == 0:
                first_error = err

        await safe_edit_message(placeholder, "All AI models and fallbacks busy. Wait 30s and try /clear")
    finally:
        stop_typing_event.set()
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

# --- Auto Register Bot Commands via post_init Hook ---
async def post_init(application):
    commands = [
        BotCommand("start", "Interactive dashboard & welcome menu"),
        BotCommand("model", "Interactive AI model picker"),
        BotCommand("imagine", "Generate Flux AI image"),
        BotCommand("ollama", "Inspect local Ollama models"),
        BotCommand("clear", "Reset conversation memory"),
        BotCommand("persona", "View or set custom personality"),
        BotCommand("persona_reset", "Reset personality to default"),
        BotCommand("auto", "Smart auto-routing"),
        BotCommand("deepseek", "DeepSeek reasoning & math"),
        BotCommand("coding", "Best coding & debugging AI"),
        BotCommand("chat", "Best fluid chat AI"),
        BotCommand("dolphin", "Dolphin 3.0 (Uncensored)"),
        BotCommand("dolmis", "Dolphin Mistral 7B (Uncensored)"),
        BotCommand("gemma", "Google Gemma 4"),
    ]
    try:
        await application.bot.set_my_commands(commands)
        logger.info("Bot commands successfully registered with Telegram.")
    except Exception as e:
        logger.warning(f"Failed to auto-register commands with Telegram: {e}")

# --- Centralized Global Error Handler ---
async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ An unexpected error occurred while processing your request. Please try again or use /clear."
            )
        except Exception:
            pass

def main():
    if not TELEGRAM_TOKEN:
        logger.error("Missing TELEGRAM_TOKEN in environment variables.")
        return

    # Setup persistent state across restarts
    persistence = PicklePersistence(filepath="bot_persistence.pickle")
    # Rate limiter to prevent HTTP 429 Flood Wait
    rate_limiter = AIORateLimiter(max_retries=3)

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .persistence(persistence)
        .rate_limiter(rate_limiter)
        .post_init(post_init)
        .connect_timeout(30.0)
        .read_timeout(60.0)
        .write_timeout(60.0)
        .build()
    )

    # Register Global Error Handler
    app.add_error_handler(global_error_handler)

    # Register Handlers
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CommandHandler("persona", persona_cmd))
    app.add_handler(CommandHandler("persona_reset", persona_reset_cmd))
    app.add_handler(CommandHandler("model", model_cmd))
    app.add_handler(CommandHandler("ollama", ollama_cmd))
    app.add_handler(CommandHandler("dolmis", dolmis_cmd))
    app.add_handler(CommandHandler("dolphin", dolphin_cmd))
    app.add_handler(CommandHandler("deepseek", deepseek_cmd))
    app.add_handler(CommandHandler("gemma", gemma_cmd))
    app.add_handler(CommandHandler("coding", coding_cmd))
    app.add_handler(CommandHandler("chat", chat_cmd))
    app.add_handler(CommandHandler("auto", auto_cmd))
    app.add_handler(CommandHandler("imagine", imagine_cmd))

    # Callback Query Handler for Interactive UI Buttons
    app.add_handler(CallbackQueryHandler(handle_callback_query))

    # Main Text Message Handler
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
