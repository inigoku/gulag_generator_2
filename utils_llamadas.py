from config import (
    GROQ_API_KEY, GROQ_MODEL, REWORK_RETRIES,
    GOOGLE_MODEL, GOOGLE_API_KEY,
    GOOGLE_MUSIC_MODEL, GOOGLE_MUSIC_VOICE,
)

import requests
import base64
import importlib

import unicodedata
import random
import json
import os
import time


# ============================
#  CLIENTE GOOGLE (SDK NUEVO)
# ============================

google_client = None
if GOOGLE_API_KEY:
    try:
        genai_module = importlib.import_module("google.genai")
        google_client = genai_module.Client(api_key=GOOGLE_API_KEY)
    except Exception:
        google_client = None



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
#  LLAMADA A GOOGLE (AUDIO)
# ============================

def _limpiar_estilo_musical(texto):
    estilo = (texto or "").strip()
    if not estilo:
        return "cinematic orchestral, emotive, dynamic build"
    estilo = estilo.replace("\n", " ").replace('"', "").strip()
    return estilo[:220]


def generar_estilo_musical_desde_contexto(contexto_poetico, poema_texto, tema="", tono_extra="", model=None):
    prompt = f"""
Eres un director musical para IA generativa de canciones.

Genera SOLO una descripción de estilo musical en una sola línea, sin comillas.
Debe estar en español y ser útil para dirigir una interpretación cantada.
Incluye: género(s), tempo aproximado, instrumentación principal, voz y atmósfera emocional.
Longitud máxima: 220 caracteres.

TEMA: {tema}
TONO EXTRA: {tono_extra}

CONTEXTO POÉTICO:
{contexto_poetico}

LETRA (poema):
{poema_texto}
"""
    estilo = llamar_google(prompt, model=model)
    return _limpiar_estilo_musical(estilo)


def consultar_estado_tarea_suno(task_id):
    raise Exception("La consulta por task_id ya no aplica: la música se genera con Google AI en una sola llamada.")


def _generar_audio_con_google(prompt_audio, model=None, voice_name=None):
    if google_client is None:
        raise Exception("Cliente de Google no inicializado")

    types_module = importlib.import_module("google.genai.types")
    selected_model = model or GOOGLE_MUSIC_MODEL
    es_lyria = "lyria" in selected_model.lower()

    if es_lyria:
        config = types_module.GenerateContentConfig(
            response_modalities=["AUDIO"],
        )
    else:
        config = types_module.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types_module.SpeechConfig(
                voice_config=types_module.VoiceConfig(
                    prebuilt_voice_config=types_module.PrebuiltVoiceConfig(
                        voice_name=voice_name or GOOGLE_MUSIC_VOICE
                    )
                )
            ),
        )

    response = google_client.models.generate_content(
        model=selected_model,
        contents=prompt_audio,
        config=config,
    )

    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                data = inline_data.data
                if isinstance(data, str):
                    return base64.b64decode(data)
                return data

    raise Exception("Google AI no devolvió audio en la respuesta")


def generar_audio_poema_con_suno(
    poema_texto,
    contexto_poetico,
    tema="",
    tono_extra="",
    model_google=None,
    estilo_musical_override="",
    titulo="",
    instrumental=False,
    wait_audio=True,
    duracion_segundos=90
):
    estilo_manual = (estilo_musical_override or "").strip()
    if estilo_manual:
        estilo_musical = _limpiar_estilo_musical(estilo_manual)
    else:
        estilo_musical = generar_estilo_musical_desde_contexto(
            contexto_poetico=contexto_poetico,
            poema_texto=poema_texto,
            tema=tema,
            tono_extra=tono_extra,
            model=model_google
        )

    titulo_final = (titulo or (f"Poema - {tema}" if tema else "Poema generado"))[:80]
    usa_lyria = "lyria" in (GOOGLE_MUSIC_MODEL or "").lower()
    modo = "instrumental" if instrumental else "con voz melódica"
    prompt_audio = f"""
Compón una pieza musical corta en formato {modo}.

TÍTULO: {titulo_final}
ESTILO MUSICAL: {estilo_musical}
TEMA: {tema}
TONO EXTRA: {tono_extra}

LETRA / TEXTO BASE:
{poema_texto}

INDICACIONES:
- Debe sonar como canción completa y coherente.
- Mantener claridad rítmica y expresividad emocional.
- Si el modelo soporta música generativa (Lyria), prioriza textura, armonía y dinámica musical.
""".strip()

    audio_bytes = _generar_audio_con_google(
        prompt_audio=prompt_audio,
        model=GOOGLE_MUSIC_MODEL,
        voice_name=GOOGLE_MUSIC_VOICE,
    )

    data = {
        "provider": "googleai",
        "model": GOOGLE_MUSIC_MODEL,
        "voice": None if usa_lyria else GOOGLE_MUSIC_VOICE,
        "wait_audio": bool(wait_audio),
        "duracion_solicitada": duracion_segundos,
    }

    return {
        "audio_url": None,
        "audio_bytes": audio_bytes,
        "task_id": None,
        "status": "completed",
        "estilo_musical": estilo_musical,
        "respuesta_suno": data
    }

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
