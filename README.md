# Personal Telegram Chatbot powered by Google Gemini

This is a personal Telegram chatbot that uses the Google Gemini API to generate responses. It maintains chat history to provide contextual conversations.

## Setup Instructions

1. **Get a Telegram Bot Token:**
   - Go to Telegram and search for `@BotFather`.
   - Send `/newbot` and follow the instructions to create a new bot.
   - Copy the API token provided.

2. **Get your Telegram User ID:**
   - To make this bot personal (only responds to you), search for `@userinfobot` or `@RawDataBot` on Telegram.
   - Start the bot to get your `id` (a number).

3. **Get a Google Gemini API Key:**
   - Go to [Google AI Studio](https://aistudio.google.com/).
   - Create a new API key.

4. **Local Configuration:**
   - Rename `.env.example` to `.env` in this directory.
   - Fill in your `TELEGRAM_BOT_TOKEN`, `GEMINI_API_KEY`, and `ALLOWED_USER_ID`.

5. **Local Testing:**
   - Install dependencies: `pip install -r requirements.txt`
   - Run the bot: `python bot.py`

## Cloud Deployment (Render)

Render is a great free option to host this bot.
1. Push this code to a GitHub repository.
2. Sign up on [Render.com](https://render.com/).
3. Click "New" and select "Web Service".
4. Connect your GitHub repository.
5. Setup the deployment:
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python bot.py`
6. Go to "Environment" variables on Render and add the variables from your `.env` file.
   - **Crucial Step:** Add `WEBHOOK_URL` and set it to your app's public Render URL (e.g., `https://your-app-name.onrender.com`). This tells the bot to use Webhooks instead of long-polling, which allows Telegram to wake up your bot instantly when you send a message!
7. Deploy!
