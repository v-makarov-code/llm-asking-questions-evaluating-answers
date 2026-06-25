import json
import os

import requests
from dotenv import load_dotenv


load_dotenv()

auth_token = os.getenv("AUTH_TOKEN")
if not auth_token:
    raise RuntimeError("AUTH_TOKEN is not set in .env")

url = "https://ai.sapiens.solutions/api/v1/conversations/ask/stream"

response = requests.post(
    url,
    headers={
        "Authorization": f"Token {auth_token}",
        "Content-Type": "application/json",
    },
    json={"query": "Какие проекты были у клиента Сбербанк?"},
    stream=True,
    timeout=300,
)
response.raise_for_status()

text = ""
for line in response.iter_lines(decode_unicode=True):
    if not line or not line.startswith("data: "):
        continue
    event = json.loads(line[6:])
    if event.get("type") == "text":
        text += event.get("data", {}).get("delta", "")

print(text)
