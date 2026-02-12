from config import (
    GROQ_API_KEY, GROQ_MODEL, REWORK_RETRIES,
    GOOGLE_MODEL, GOOGLE_API_KEY
)

import requests
import base64
import google.genai as genai

import unicodedata
import random
import json
import os


# ============================
#  CLIENTE GOOGLE (SDK NUEVO)
# ============================

google_client = None
if GOOGLE_API_KEY:
    google_client = genai.Client(api_key=GOOGLE_API_KEY)



# ============================
#  UTILIDADES
# ============================

def limpiar_prompt(texto):
    texto = unicodedata.normalize("NFC", texto)
    return texto.strip()



# ============================
#  LLAMADA A GROQ
# ============================

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



# ============================
#  LLAMADA A GOOGLE (TEXTO)
# ============================

def llamar_google(prompt, system_prompt=None, model=None):
    if not GOOGLE_API_KEY:
        raise Exception("Google API Key no configurada")

    if google_client is None:
        raise Exception("Cliente de Google no inicializado")

    print("=== DEBUG (GOOGLE AI STUDIO) ===")
    print("MODEL:", model or GOOGLE_MODEL)

    try:
        final_prompt = prompt
        if system_prompt:
            final_prompt = f"INSTRUCCIONES DEL SISTEMA:\n{system_prompt}\n\n---\n\n{prompt}"

        print(f"--- FULL PROMPT (GOOGLE) ---\n{final_prompt}\n----------------------------")

        response = google_client.models.generate_content(
            model=model or GOOGLE_MODEL,
            contents=final_prompt
        )

        return response.text

    except Exception as e:
        raise Exception(f"Error llamando a Google AI Studio: {e}")



# ============================
#  LLAMADA A GOOGLE (IMAGEN)
# ============================

def generate_from_poem(poem_text):
    """
    Uses Gemini 2.5 Flash Image to visualize a poem.
    """
    if not GOOGLE_API_KEY:
        raise Exception("Google API Key no configurada")
    try:
        # 3. Define the prompt
        prompt = f'Create a highly artistic, atmospheric, and detailed visual representation inspired by this poem: "{poem_text}". Focus on the emotional resonance and symbolism.'

        # 4. Generate Content
        response = google_client.models.generate_content(
            model='gemini-2.5-flash-image',
            contents=prompt
        )
        # 5. Extract and display the image
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                return part.inline_data.data
            
        return None

    except Exception as e:
        raise Exception(f"Error llamando a Google Image API (REST): {e}")

def guardar_imagen(base64_data, ruta="imagen.png"):
    with open(ruta, "wb") as f:
        f.write(base64.b64decode(base64_data))
    return ruta

# ============================
#  UTILIDADES DE ARCHIVOS
# ============================

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
