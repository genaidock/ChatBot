#!/bin/bash
set -e

echo "============================================="
echo "=== Updating Telegram Bot on Oracle Server =="
echo "============================================="

BOT_DIR="/home/ubuntu/telegram-bot"

if [ ! -d "$BOT_DIR" ]; then
    echo "Directory $BOT_DIR does not exist. Creating..."
    mkdir -p "$BOT_DIR"
fi

cd "$BOT_DIR"

# 1. Check Ollama Models
echo ""
echo "--- Checking Ollama Models on Oracle ---"
if command -v ollama >/dev/null 2>&1; then
    echo "Ollama is installed. Checking models..."
    ollama list || true
    echo ""
    echo "Tip: If dolphin models are missing, run:"
    echo "  ollama pull dolphin3"
    echo "  ollama pull dolphin-mistral:7b"
else
    echo "⚠️ Warning: ollama command not found in PATH."
fi

# 2. Stop running service
echo ""
echo "--- Stopping telegram-bot service ---"
sudo systemctl stop telegram-bot || true

# 3. Create backup of current bot.py and state
if [ -f "$BOT_DIR/bot.py" ]; then
    cp "$BOT_DIR/bot.py" "$BOT_DIR/bot.py.bak.$(date +%F_%T)"
    echo "Backup created at bot.py.bak.*"
fi
if [ -f "$BOT_DIR/bot_persistence.pickle" ]; then
    cp "$BOT_DIR/bot_persistence.pickle" "$BOT_DIR/bot_persistence.pickle.bak.$(date +%F_%T)"
    echo "State backup created at bot_persistence.pickle.bak.*"
fi

# 4. Fetch latest bot.py and requirements
echo ""
echo "--- Ensuring system audio libraries ---"
sudo apt-get update -qq && sudo apt-get install -y -qq git libsndfile1 || true

echo "--- Updating code ---"
if [ -d "$BOT_DIR/.git" ]; then
    git pull origin main || true
else
    echo "Downloading latest bot.py from GitHub..."
    curl -fsSL https://raw.githubusercontent.com/genaidock/ChatBot/main/bot.py -o "$BOT_DIR/bot.py"
    curl -fsSL https://raw.githubusercontent.com/genaidock/ChatBot/main/requirements.txt -o "$BOT_DIR/requirements.txt"
fi

# 5. Activate virtual environment and install dependencies
echo ""
echo "--- Installing dependencies in venv ---"
if [ -d "$BOT_DIR/venv" ]; then
    source "$BOT_DIR/venv/bin/activate"
else
    echo "Creating virtual environment at $BOT_DIR/venv..."
    python3 -m venv "$BOT_DIR/venv"
    source "$BOT_DIR/venv/bin/activate"
fi

pip install --upgrade pip
pip install --upgrade -r "$BOT_DIR/requirements.txt"

# 6. Verify .env settings
if [ -f "$BOT_DIR/.env" ]; then
    # Ensure localhost:11434 is used for systemd deployment
    sed -i 's|http://host.docker.internal:11434|http://localhost:11434|g' "$BOT_DIR/.env"
fi

# 7. Restart service
echo ""
echo "--- Restarting telegram-bot service ---"
sudo systemctl restart telegram-bot

# 8. Check status
echo ""
echo "--- Service Status ---"
sudo systemctl status telegram-bot --no-pager

echo ""
echo "============================================="
echo "=== Update Complete! Check logs with:       =="
echo "=== sudo journalctl -u telegram-bot -f     =="
echo "============================================="
