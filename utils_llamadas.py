
from config import (
    GROQ_API_KEY, GROQ_MODEL, REWORK_RETRIES,
    GOOGLE_MODEL, GOOGLE_API_KEY
)

import requests
from google import genai

import unicodedata
import random
import json
import os

# Configuración para Google AI Studio (Capa gratuita sin Vertex)
if GOOGLE_API_KEY:
    from google import genai
    client = genai.Client(api_key=GOOGLE_API_KEY)

def limpiar_prompt(texto):
    # Normaliza unicode (NFC) pero mantiene caracteres latinos como tildes y ñ
    texto = unicodedata.normalize("NFC", texto)
    return texto.strip()

def llamar_groq(prompt, system_prompt="Eres un asistente experto en poesía generativa.", model=None):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    payload = {
        "model": model or GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.9,
        "max_tokens": 1600
    }

    print("=== DEBUG (GROQ) ===")
    print("MODEL:", payload["model"])
    print("API KEY:", "OK" if GROQ_API_KEY else "MISSING")
    print(f"--- SYSTEM PROMPT ---\n{system_prompt}\n---------------------")
    print(f"--- USER PROMPT ---\n{prompt}\n-------------------")
    print("PAYLOAD:", payload)


    import time
    max_retries = REWORK_RETRIES
    for intento in range(max_retries):
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 429:
            print("Rate limit alcanzado. Esperando 10 segundos antes de reintentar...")
            time.sleep(10)
            continue
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    raise Exception("Demasiados intentos fallidos por rate limit (429)")

def llamar_google(prompt, system_prompt=None, model=None):
    if GOOGLE_API_KEY:
        print("=== DEBUG (GOOGLE AI STUDIO) ===")
        
        print("MODEL:", model or GOOGLE_MODEL)
        try:
            final_prompt = prompt
            if system_prompt:
                final_prompt = f"INSTRUCCIONES DEL SISTEMA:\n{system_prompt}\n\n---\n\n{prompt}"

            print(f"--- FULL PROMPT (GOOGLE) ---\n{final_prompt}\n----------------------------")
            response = client.models.generate_content(
                model=model or GOOGLE_MODEL,
                contents=final_prompt
            )
            return response.text
        except Exception as e:
            raise Exception(f"Error llamando a Google AI Studio: {e}")
    raise Exception("Google API Key no configurada")

def leer_texto(ruta):
    if os.path.exists(ruta):
        with open(ruta, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""

def cargar_prompt(ruta):
    return leer_texto(ruta)

def cargar_json(ruta):
    if os.path.exists(ruta):
        with open(ruta, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def seleccionar(lista, k):
    if not lista:
        return []
    return random.sample(lista, min(len(lista), k))
