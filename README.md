# Telegram MetaBot (Multi-Provider AI & Image Generation)

A Telegram chatbot featuring multi-provider streaming fallbacks, uncensored Dolphin-style personas, custom personality switching, and free AI image generation via Flux.

## Key Features

- **Multi-Provider Fallback Cascade:**
  1. **Venice AI**: `dolphin-mistral-24b-venice-edition`
  2. **OpenRouter**: `cognitivecomputations/dolphin-mistral-24b-venice-edition:free`
  3. **Pollinations.ai**: `openai` (100% free fallback, no API key required)
- **Real-Time Streaming Responses:** Smooth streamed output with live typing indicators.
- **Custom Personas:** Set any custom persona on the fly per chat (`/persona <prompt>`), or reset to default MetaBot (`/persona_reset`).
- **AI Image Generation:** Free, high-quality 1024x1024 Flux image generation via `/imagine <prompt>`.
- **Group & Private Chat Support:** Works automatically in private messages; responds in groups when @mentioned, replied to, or on commands.
- **Dual Polling & Webhook Support:** Seamless local development with polling; automatic webhook mode for 24/7 cloud hosting on platforms like Render.

## Bot Commands

- `/start` - Welcome message and command list
- `/clear` - Clear chat context memory
- `/persona <text>` - Set a custom personality for this chat
- `/persona_reset` - Reset persona back to default MetaBot
- `/model` - Display active providers and priority cascade
- `/imagine <prompt>` - Generate an image using Flux

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
