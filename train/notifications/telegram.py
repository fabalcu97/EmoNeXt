import requests

from os import getenv
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_API_KEY = getenv("TELEGRAM_API_KEY")
TELEGRAM_CHAT_ID = getenv("TELEGRAM_CHAT_ID")


class TelegramNotifier:
    def __init__(self, api_token: str = TELEGRAM_API_KEY, chat_id: str = TELEGRAM_CHAT_ID):
        self.api_token = api_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.api_token}/sendMessage"

    def send_message(self, message: str) -> bool:
        payload = {"chat_id": self.chat_id, "text": message}
        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"Failed to send message: {e}")
            return False
