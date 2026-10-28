#!/bin/bash
set -e

echo "=== Updating Telegram Bot on Oracle Server ==="

BOT_DIR="/home/ubuntu/telegram-bot"
cd "$BOT_DIR"

# 1. Stop running service
echo "Stopping telegram-bot service..."
sudo systemctl stop telegram-bot

# 2. Activate virtual environment and install dependencies
echo "Installing/updating required Python packages..."
source "$BOT_DIR/venv/bin/activate"
pip install --upgrade openai python-dotenv "python-telegram-bot[webhooks]"

# 3. Create backup of current bot.py
if [ -f "$BOT_DIR/bot.py" ]; then
    cp "$BOT_DIR/bot.py" "$BOT_DIR/bot.py.bak.$(date +%F_%T)"
    echo "Backup created at bot.py.bak.*"
fi

# 4. Restart service
echo "Restarting telegram-bot service..."
sudo systemctl restart telegram-bot

# 5. Check status
echo "Checking service status..."
sudo systemctl status telegram-bot --no-pager

echo "=== Update Complete! ==="
