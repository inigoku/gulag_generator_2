from config import (
    GROQ_API_KEY, GROQ_MODEL, REWORK_RETRIES,
    GOOGLE_MODEL, GOOGLE_API_KEY,
    SUNO_API_KEY, SUNO_API_URL,
    SUNO_MODEL,
    SUNO_CALLBACK_URL,
    SUNO_STATUS_URL, SUNO_POLL_ATTEMPTS, SUNO_POLL_INTERVAL_SEC
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
#  LLAMADA A SUNO (AUDIO)
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
Debe estar en español y ser útil para un generador musical tipo Suno.
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


def _extraer_audio_url(data):
    if not isinstance(data, dict):
        return None

    def recolectar_candidatos(valor, bolsa):
        if isinstance(valor, dict):
            for k, v in valor.items():
                clave = str(k).lower()
                if isinstance(v, str) and (
                    "audio" in clave or
                    clave in {"url", "song_url", "clip_url", "media_url", "source_audio_url", "stream_audio_url"}
                ):
                    bolsa.append(v)
                recolectar_candidatos(v, bolsa)
        elif isinstance(valor, list):
            for item in valor:
                recolectar_candidatos(item, bolsa)

    candidatos = []
    recolectar_candidatos(data, candidatos)

    for valor in candidatos:
        if isinstance(valor, str) and valor.startswith("http"):
            return valor

    return None


def _extraer_task_id(data):
    if not isinstance(data, dict):
        return None

    claves_task = {"task_id", "taskid", "job_id", "jobid", "request_id", "requestid", "task"}
    pila = [data]

    while pila:
        actual = pila.pop()
        if isinstance(actual, dict):
            for k, v in actual.items():
                k_norm = str(k).lower().replace("-", "_")
                if k_norm in claves_task and isinstance(v, (str, int)):
                    return str(v)
                if isinstance(v, (dict, list)):
                    pila.append(v)
        elif isinstance(actual, list):
            pila.extend(actual)

    return None


def _extraer_status(data):
    if not isinstance(data, dict):
        return None

    claves = {"status", "state", "task_status", "job_status"}
    pila = [data]
    while pila:
        actual = pila.pop()
        if isinstance(actual, dict):
            for k, v in actual.items():
                if str(k).lower() in claves and isinstance(v, str):
                    return v
                if isinstance(v, (dict, list)):
                    pila.append(v)
        elif isinstance(actual, list):
            pila.extend(actual)
    return None


def _construir_status_urls(task_id):
    urls = []
    base_generate = (SUNO_API_URL or "").rstrip("/")
    base_status = (SUNO_STATUS_URL or "").strip()

    if base_status:
        if "{task_id}" in base_status:
            urls.append(base_status.replace("{task_id}", task_id))
        else:
            urls.extend([
                f"{base_status.rstrip('/')}/{task_id}",
                f"{base_status}?taskId={task_id}",
                f"{base_status}?task_id={task_id}",
                f"{base_status}?id={task_id}"
            ])

    if base_generate:
        urls.extend([
            f"{base_generate}/{task_id}",
            f"{base_generate}?taskId={task_id}",
            f"{base_generate}?task_id={task_id}",
            f"{base_generate}?id={task_id}",
            f"{base_generate}/status/{task_id}",
            f"{base_generate}/record-info?taskId={task_id}",
            f"{base_generate}/record-info?task_id={task_id}",
            f"{base_generate}/record-info?id={task_id}"
        ])

    dedup = []
    seen = set()
    for u in urls:
        if u not in seen:
            dedup.append(u)
            seen.add(u)
    return dedup


def _poll_suno_hasta_audio(task_id, headers):
    urls = _construir_status_urls(task_id)
    if not urls:
        return None, None, None

    ultimo_data = None
    ultimo_status = None

    for _ in range(max(1, SUNO_POLL_ATTEMPTS)):
        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code >= 400:
                    continue
                data = resp.json() if resp.content else {}
                if isinstance(data, dict) and int(data.get("code", 200)) != 200:
                    continue
                ultimo_data = data
                audio_url = _extraer_audio_url(data)
                status = _extraer_status(data)
                ultimo_status = status
                if audio_url:
                    return audio_url, status, data
                if status and str(status).lower() in {"failed", "error", "cancelled"}:
                    return None, status, data
            except Exception:
                continue
        time.sleep(max(1, SUNO_POLL_INTERVAL_SEC))

    return None, ultimo_status, ultimo_data


def consultar_estado_tarea_suno(task_id):
    if not SUNO_API_KEY:
        raise Exception("SUNO_API_KEY no configurada")
    if not task_id:
        raise Exception("task_id vacío")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SUNO_API_KEY}",
        "X-API-KEY": SUNO_API_KEY
    }

    urls = _construir_status_urls(str(task_id).strip())
    ultimo_error = None
    ultimo_data = None

    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code >= 400:
                ultimo_error = f"HTTP {resp.status_code} en {url}"
                continue

            data = resp.json() if resp.content else {}
            if isinstance(data, dict) and int(data.get("code", 200)) != 200:
                ultimo_error = f"API code={data.get('code')} msg={data.get('msg')} en {url}"
                continue
            ultimo_data = data
            audio_url = _extraer_audio_url(data)
            status = _extraer_status(data)
            found_task_id = _extraer_task_id(data) or str(task_id).strip()

            return {
                "task_id": found_task_id,
                "status": status or "unknown",
                "audio_url": audio_url,
                "respuesta_suno": data,
                "consulta_url": url
            }
        except Exception as e:
            ultimo_error = str(e)

    if ultimo_data is not None:
        return {
            "task_id": str(task_id).strip(),
            "status": _extraer_status(ultimo_data) or "unknown",
            "audio_url": _extraer_audio_url(ultimo_data),
            "respuesta_suno": ultimo_data,
            "consulta_url": None
        }

    raise Exception(f"No se pudo consultar estado para task_id={task_id}. Último error: {ultimo_error}")


def generar_audio_poema_con_suno(
    poema_texto,
    contexto_poetico,
    tema="",
    tono_extra="",
    model_google=None,
    titulo="",
    instrumental=False,
    wait_audio=True,
    duracion_segundos=90
):
    if not SUNO_API_KEY:
        raise Exception("SUNO_API_KEY no configurada")
    if not SUNO_API_URL:
        raise Exception("SUNO_API_URL no configurada")

    estilo_musical = generar_estilo_musical_desde_contexto(
        contexto_poetico=contexto_poetico,
        poema_texto=poema_texto,
        tema=tema,
        tono_extra=tono_extra,
        model=model_google
    )

    titulo_final = (titulo or (f"Poema - {tema}" if tema else "Poema generado"))[:80]
    duracion_limpia = int(duracion_segundos) if duracion_segundos else 90
    if duracion_limpia < 30:
        duracion_limpia = 30
    if duracion_limpia > 240:
        duracion_limpia = 240

    payload = {
        "prompt": estilo_musical,
        "style": estilo_musical,
        "lyrics": poema_texto,
        "text": poema_texto,
        "title": titulo_final,
        "model": SUNO_MODEL,
        "customMode": True,
        "instrumental": bool(instrumental),
        "make_instrumental": bool(instrumental),
        "wait_audio": bool(wait_audio),
        "duration": duracion_limpia,
        "duration_seconds": duracion_limpia
    }

    if SUNO_CALLBACK_URL:
        payload["callBackUrl"] = SUNO_CALLBACK_URL
        payload["callbackUrl"] = SUNO_CALLBACK_URL

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SUNO_API_KEY}",
        "X-API-KEY": SUNO_API_KEY
    }

    response = requests.post(SUNO_API_URL, headers=headers, json=payload, timeout=120)
    if response.status_code >= 400:
        raise Exception(f"Error Suno API ({response.status_code}): {response.text[:300]}")

    data = response.json() if response.content else {}
    msg = str(data.get("msg", "")) if isinstance(data, dict) else ""
    if isinstance(data, dict) and data.get("code") and int(data.get("code", 0)) != 200:
        if "callbackurl" in msg.lower() and not SUNO_CALLBACK_URL:
            raise Exception(
                "SunoAPI requiere SUNO_CALLBACK_URL en .env (campo callBackUrl). "
                "Configura una URL pública de webhook y reintenta."
            )
        raise Exception(f"Error Suno API ({data.get('code')}): {msg}")

    audio_url = _extraer_audio_url(data)
    status = _extraer_status(data)
    task_id = _extraer_task_id(data)

    if not audio_url and task_id:
        audio_url, poll_status, poll_data = _poll_suno_hasta_audio(task_id, headers)
        if poll_data is not None:
            data = poll_data
        if poll_status:
            status = poll_status

    if not audio_url and task_id:
        return {
            "audio_url": None,
            "task_id": task_id,
            "status": status or "processing",
            "estilo_musical": estilo_musical,
            "respuesta_suno": data
        }

    if not audio_url:
        snippet = str(data)[:350]
        raise Exception(f"Suno API respondió sin URL de audio ni task_id reconocible. Respuesta parcial: {snippet}")

    return {
        "audio_url": audio_url,
        "task_id": task_id,
        "status": status or "completed",
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
