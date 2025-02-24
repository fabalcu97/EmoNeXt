import requests


class TelegramNotifier:
    def __init__(self, api_token: str, chat_id: str):
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
