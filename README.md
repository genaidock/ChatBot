# Telegram MetaBot (Multi-Provider AI & Image Generation)

A Telegram chatbot featuring multi-provider streaming fallbacks, uncensored Dolphin-style personas, custom personality switching, and free AI image generation via Flux.

## Key Features

- **Interactive Inline UI**: Dynamic model selection dashboard and priority toggles with Inline Keyboards & Callback Queries (`/model`, `/start`).
- **Persistent State Across Restarts**: Preserves chat memory, selected models, and custom personas using `PicklePersistence` across systemd restarts and redeployments.
- **Auto-registered Command Autocompletion**: Auto-registers the bot command catalog via Telegram's `set_my_commands` API during startup (`post_init`).
- **Telegram Rate-Limiting Protection**: Integrates `AIORateLimiter` to protect against HTTP 429 Flood Wait rate limits during streaming.
- **Continuous Typing Indicators**: Active typing indicator keeps users informed during long Ollama cold starts.
- **Safe Markdown Rendering**: Resilient entity parser that automatically falls back to raw text if an LLM produces unclosed Markdown symbols.
- **Multi-Provider Fallback Cascade:**
  1. **OmniRoute Gateway / Local Ollama (Oracle VPS)**
  2. **Venice AI**: Uncensored cloud models
  3. **OpenRouter**: High-performance reasoning models
  4. **Pollinations.ai**: `openai` (100% free safety net fallback)
- **Real-Time Streaming Responses:** Smooth streamed output with live edit throttle.
- **Text-to-Speech (Paradee-8M):** Ultra-lightweight local CPU-driven voice synthesis (distilled from Kokoro-82M) via `/speak` or reply-to-speak.
- **Custom Personas:** Set any custom persona on the fly per chat (`/persona <prompt>`), or reset to default MetaBot (`/persona_reset`).
- **AI Image Generation:** Free, high-quality Flux image generation via `/imagine <prompt>`.
- **Group & Private Chat Support:** Works automatically in private messages; responds in groups when @mentioned, replied to, or on commands.
- **Dual Polling & Webhook Support:** Seamless local/systemd development with polling; automatic webhook mode for 24/7 cloud hosting on platforms like Render.

## Bot Commands

- `/start` - Interactive welcome dashboard and menu
- `/model` - Interactive model dashboard and cascade switcher
- `/ollama` - Check local Ollama status & downloaded models
- `/imagine <prompt>` - Generate an image using Flux AI
- `/edit <prompt>` - Reply to any photo to transform/edit with AI (Img2Img)
- `/search <query>` - Live web search with AI synthesized answer
- `/speak <text>` - Voice synthesis via Paradee 8M (or reply to any message with `/speak`)
- `/voice` - Alias for `/speak` (supports `heart`, `puck`, `isabella`, `fable`)
- `/clear` - Reset chat context memory
- `/persona <text>` - Set a custom personality for this chat
- `/persona_reset` - Reset persona back to default MetaBot
- `/auto` - Smart auto-routing
- `/deepseek` - Switch to DeepSeek
- `/coding` - Switch to Best Coding AI
- `/chat` - Switch to Best Chat AI
- `/dolphin` - Switch to Dolphin 3.0 (Uncensored)
- `/dolmis` - Switch to Dolphin Mistral 7B (Uncensored)
- `/gemma` - Switch to Google Gemma 4

## Setup Instructions

### 1. Environment Configuration

Create or update `.env` in the root directory:

```env
TELEGRAM_TOKEN=your_telegram_bot_token_here
VENICE_KEY=your_venice_api_key_here          # Optional, falls back to OpenRouter/Pollinations
OPENROUTER_KEY=your_openrouter_api_key_here  # Optional, falls back to Pollinations
# WEBHOOK_URL=https://your-app-name.onrender.com  # Required only for Webhook deployment on Render
```

*(Note: `TELEGRAM_BOT_TOKEN` is also supported for backward compatibility)*

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Locally

```bash
python bot.py
```

## Cloud Deployment (Render)

1. Push your changes to your GitHub repository:
   ```bash
   git add .
   git commit -m "feat: multi-provider AI bot with image gen and personas"
   git push
   ```
2. On **Render.com**, configure your Web Service:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python bot.py`
3. Under **Environment Variables**, set:
   - `TELEGRAM_TOKEN` (or `TELEGRAM_BOT_TOKEN`)
   - `VENICE_KEY` (optional)
   - `OPENROUTER_KEY` (optional)
   - `WEBHOOK_URL` set to `https://your-service-name.onrender.com`
