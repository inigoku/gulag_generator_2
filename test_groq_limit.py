import requests
import json
import os
import sys

from config import get_config

API_KEY = get_config("GROQ_API_KEY", "")
if not API_KEY:
    print("No GROQ API KEY found")
    sys.exit(1)

models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"]

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

for model in models:
    # We will test payload sizes
    for size_kb in [20, 25, 30, 40, 50, 100, 120]:
        size_bytes = size_kb * 1024
        
        # create a prompt of this size roughly
        prompt = "A" * size_bytes
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10
        }
        
        payload_bytes = len(json.dumps(payload).encode('utf-8'))
        
        try:
            res = requests.post(url, headers=headers, json=payload)
            if res.status_code == 413:
                print(f"Model: {model} | Size: {payload_bytes} bytes -> 413 Payload Too Large")
                break
            else:
                print(f"Model: {model} | Size: {payload_bytes} bytes -> {res.status_code}")
        except Exception as e:
            print(f"Error: {e}")
            break
