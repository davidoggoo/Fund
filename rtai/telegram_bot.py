"""Telegram bot integration for RTAI trading signals"""

import asyncio
import os
from typing import Optional
import httpx
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


class TelegramBot:
    """Simple Telegram bot for sending trading signals"""
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if not self.bot_token:
            logger.warning("Telegram disabled: TELEGRAM_BOT_TOKEN not set")
            self.enabled = False
        elif not self.chat_id:
            logger.warning("Telegram disabled: TELEGRAM_CHAT_ID not set")
            self.enabled = False
        else:
            logger.info(f"📱 Telegram bot initialized for chat: {self.chat_id}")
    
    async def send_message(self, text: str) -> bool:
        """Send a message to Telegram"""
        if not self.enabled:
            logger.debug(f"📱 [TELEGRAM DISABLED] {text}")
            return False
            
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=data, timeout=10)
                response.raise_for_status()
                print(f"📱 Telegram sent: {text}")
                return True
                
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            return False
    
    async def send_signal(self, symbol: str, signal_type: str, price: float, reason: str, strength: float):
        """Send a trading signal with enhanced formatting"""
        emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "⚡"
        
        # Format strength as percentage
        strength_pct = min(strength * 100, 100)
        
        message = f"""
{emoji} <b>RTAI EXTREME SIGNAL</b>

<b>Symbol:</b> {symbol}
<b>Signal:</b> {signal_type}
<b>Price:</b> ${price:,.2f}
<b>Strength:</b> {strength_pct:.0f}%
<b>Reason:</b> {reason}
<b>Time:</b> {time.strftime('%H:%M:%S UTC', time.gmtime())}
        """.strip()
        
        await self.send_message(message)
    
    async def stop(self):
        """Stop the Telegram bot gracefully"""
        if self.enabled:
            logger.info("📱 Telegram bot stopped")
        else:
            logger.debug("📱 Telegram bot was not running")


# Global telegram bot instance
telegram_bot = TelegramBot()


async def send_trading_signal(symbol: str, signal_type: str, indicator: str, value: float, reason: str):
    """Send a trading signal via Telegram"""
    await telegram_bot.send_signal(symbol, signal_type, indicator, value, reason)
