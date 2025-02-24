from typing import Optional

from telegram import Bot
from telegram.error import TelegramError


class TelegramNotifier:
    def __init__(self, api_key: Optional[str] = None, chat_id: Optional[str] = None):
        """Initialize the TelegramNotifier.

        Args:
            api_key: Bot API key. If None, uses TELEGRAM_API_KEY from env
            chat_id: Chat ID. If None, uses TELEGRAM_CHAT_ID from env
        """
        self.api_key = api_key
        self.chat_id = chat_id

        if not self.api_key:
            raise ValueError(
                "Telegram API key not provided and not found in environment"
            )
        if not self.chat_id:
            raise ValueError(
                "Telegram chat ID not provided and not found in environment"
            )

        self.bot = Bot(token=self.api_key)

    async def send_message(self, message: str) -> bool:
        """Send a message to the configured Telegram group.

        Args:
            message: The message to send

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="HTML",  # Enables HTML formatting
            )
            logger.info(f"Message sent successfully to chat {self.chat_id}")
            return True
        except TelegramError as e:
            logger.error(f"Failed to send message: {str(e)}")
            return False


async def main():
    """Example usage of the TelegramNotifier."""
    notifier = TelegramNotifier()
    await notifier.send_message("👋 Hello! This is a test message from the bot.")


if __name__ == "__main__":
    # Run the example
    import asyncio

    asyncio.run(main())
