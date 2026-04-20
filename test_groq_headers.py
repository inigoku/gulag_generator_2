import requests
import os
import sys

from config import get_config

API_KEY = get_config("GROQ_API_KEY", "")
url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"]

for model in models:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10
    }
    
    res = requests.post(url, headers=headers, json=payload)
    print(f"\nModel: {model}")
    print("x-ratelimit-limit-tokens:", res.headers.get("x-ratelimit-limit-tokens"))
    print("x-ratelimit-remaining-tokens:", res.headers.get("x-ratelimit-remaining-tokens"))
    print("x-ratelimit-limit-requests:", res.headers.get("x-ratelimit-limit-requests"))
