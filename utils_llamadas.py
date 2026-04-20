from config import (
    GROQ_API_KEY, GROQ_MODEL, REWORK_RETRIES,
    GOOGLE_MODEL, GOOGLE_API_KEY, GOOGLE_FAST_MODEL,
    GOOGLE_MUSIC_MODEL, GOOGLE_MUSIC_VOICE,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL
)

import requests
import base64
import importlib
import unicodedata
import random
import json
import os
import time

GROQ_MAX_PAYLOAD_BYTES = 120000
GROQ_MAX_SYSTEM_CHARS = 4000
GROQ_MAX_PROMPT_CHARS = 32000
CHARS_PER_TOKEN_APPROX = 4
MIN_INPUT_BUDGET_TOKENS = 2048

MODEL_CONTEXT_LIMITS = {
    # Groq
    "llama-3.3-70b-versatile": 16384,
    "llama-3.1-8b-instant": 16384,
    "openai/gpt-oss-20b": 16384,
    "openai/gpt-oss-120b": 16384,
    "qwen/qwen3-32b": 16384,
    # DeepSeek
    "deepseek-chat": 65536,
    # Google (aproximados conservadores por familia)
    "gemini-pro-latest": 1048576,
    "gemini-flash-latest": 1048576,
    "models/gemini-2.5-pro": 1048576,
    "models/gemini-2.5-flash": 1048576,
    "models/gemini-2.5-flash-image": 32768,
}

PROVIDER_DEFAULT_CONTEXT_LIMITS = {
    "Groq": 131072,
    "DeepSeek": 65536,
    "Google": 32768,
}

GROQ_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
]

GROQ_CALL_HISTORY = []
LLM_CALL_HISTORY = []
MODEL_ATTEMPT_HISTORY = []

# ============================
#  CLIENTE GOOGLE (SDK NUEVO)
# ============================

genai_module = None
if GOOGLE_API_KEY:
    try:
        genai_module = importlib.import_module("google.generativeai")
        genai_module.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"Error al configurar el SDK de Google: {e}")
        genai_module = None

# ============================
#  LLAMADA A GROQ
# ============================

def _construir_mensajes_chat(prompt, system_prompt=None):
    mensajes = []
    if system_prompt:
        mensajes.append({"role": "system", "content": system_prompt})
    mensajes.append({"role": "user", "content": prompt})
    return mensajes

def reset_groq_call_history():
    GROQ_CALL_HISTORY.clear()
    LLM_CALL_HISTORY.clear()
    MODEL_ATTEMPT_HISTORY.clear()

def get_groq_call_history():
    return [dict(item) for item in GROQ_CALL_HISTORY]

def get_model_attempt_history():
    return [dict(item) for item in MODEL_ATTEMPT_HISTORY]

def _registrar_llamada_llm(proveedor, modelo, solicitado=None):
    info = {
        "proveedor": proveedor,
        "modelo": modelo,
        "solicitado": solicitado or modelo,
        "label": f"{proveedor}: {modelo}",
    }
    LLM_CALL_HISTORY.append(info)
    return info

def get_last_llm_call():
    if not LLM_CALL_HISTORY:
        return None
    return dict(LLM_CALL_HISTORY[-1])

def _error_corto(error, limite=260):
    texto = str(error or "").replace("\n", " ").strip()
    if len(texto) <= limite:
        return texto
    return f"{texto[:limite]}..."

def _registrar_intento_modelo(proveedor, modelo, estado, error=None, fase=None):
    MODEL_ATTEMPT_HISTORY.append({
        "proveedor": proveedor,
        "modelo": modelo,
        "estado": estado,
        "fase": fase or "",
        "error": _error_corto(error) if error else None,
    })

def _strip_think_tags(texto):
    """Elimina bloques <think>...</think> que emiten algunos modelos (DeepSeek, Qwen)."""
    import re
    return re.sub(r"<think>.*?</think>", "", texto or "", flags=re.DOTALL).strip()

def _recortar_texto(texto, limite, etiqueta):
    if not texto or len(texto) <= limite:
        return texto

    margen = max(200, (limite - len(etiqueta) - 32) // 2)
    inicio = texto[:margen]
    final = texto[-margen:]
    return f"{inicio}\n\n[{etiqueta}: contenido recortado]\n\n{final}"

def _resumen_debug(texto, limite=1200):
    if not texto or len(texto) <= limite:
        return texto
    return f"{texto[:limite]}\n... [truncado para debug: {len(texto) - limite} caracteres omitidos]"

def _normalizar_modelo(modelo):
    return (modelo or "").strip().lower()

def _resolver_limite_contexto_modelo(modelo, proveedor):
    normalizado = _normalizar_modelo(modelo)
    if not normalizado:
        return PROVIDER_DEFAULT_CONTEXT_LIMITS.get(proveedor, 32768)

    if normalizado in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[normalizado]

    for key, valor in MODEL_CONTEXT_LIMITS.items():
        if normalizado.endswith(key):
            return valor

    return PROVIDER_DEFAULT_CONTEXT_LIMITS.get(proveedor, 32768)

def _ajustar_prompt_a_limite_modelo(prompt, system_prompt, modelo, proveedor, max_tokens_salida=1600):
    prompt_original = prompt or ""
    system_original = system_prompt or ""

    contexto_total = _resolver_limite_contexto_modelo(modelo, proveedor)
    reserva_salida = max(max_tokens_salida + 512, int(contexto_total * 0.2))
    presupuesto_entrada_tokens = max(MIN_INPUT_BUDGET_TOKENS, contexto_total - reserva_salida)
    presupuesto_entrada_chars = presupuesto_entrada_tokens * CHARS_PER_TOKEN_APPROX

    total_chars = len(prompt_original) + len(system_original)
    if total_chars <= presupuesto_entrada_chars:
        return prompt_original, system_original, {
            "modelo": modelo,
            "proveedor": proveedor,
            "context_window_tokens": contexto_total,
            "input_budget_tokens": presupuesto_entrada_tokens,
            "prompt_original_chars": len(prompt_original),
            "prompt_enviado_chars": len(prompt_original),
            "system_original_chars": len(system_original),
            "system_enviado_chars": len(system_original),
            "recortado_por_limite_modelo": False,
        }

    # Reparto de presupuesto: priorizar prompt (80%) y reservar 20% para system
    system_budget = min(len(system_original), max(1200, int(presupuesto_entrada_chars * 0.2)))
    prompt_budget = max(2000, presupuesto_entrada_chars - system_budget)

    system_ajustado = _recortar_texto(system_original, system_budget, "SYSTEM") if system_original else system_original
    prompt_ajustado = _recortar_texto(prompt_original, prompt_budget, "PROMPT")

    info = {
        "modelo": modelo,
        "proveedor": proveedor,
        "context_window_tokens": contexto_total,
        "input_budget_tokens": presupuesto_entrada_tokens,
        "prompt_original_chars": len(prompt_original),
        "prompt_enviado_chars": len(prompt_ajustado or ""),
        "system_original_chars": len(system_original),
        "system_enviado_chars": len(system_ajustado or ""),
        "recortado_por_limite_modelo": (prompt_ajustado != prompt_original) or (system_ajustado != system_original),
    }
    return prompt_ajustado, system_ajustado, info

def _preparar_payload_groq(modelo, prompt, system_prompt, max_tokens=1600):
    prompt_original = prompt or ""
    system_original = system_prompt or ""
    prompt_ajustado = _recortar_texto(prompt, GROQ_MAX_PROMPT_CHARS, "PROMPT")
    system_ajustado = _recortar_texto(system_prompt, GROQ_MAX_SYSTEM_CHARS, "SYSTEM")

    payload = {
        "model": modelo,
        "messages": _construir_mensajes_chat(prompt_ajustado, system_ajustado),
        "temperature": 0.9,
        "max_tokens": max_tokens,
    }

    while len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > GROQ_MAX_PAYLOAD_BYTES:
        if prompt_ajustado and len(prompt_ajustado) > 4000:
            prompt_ajustado = _recortar_texto(prompt_ajustado, max(4000, len(prompt_ajustado) - 4000), "PROMPT")
        elif system_ajustado and len(system_ajustado) > 1000:
            system_ajustado = _recortar_texto(system_ajustado, max(1000, len(system_ajustado) - 1000), "SYSTEM")
        else:
            break

        payload["messages"] = _construir_mensajes_chat(prompt_ajustado, system_ajustado)

    info = {
        "modelo": modelo,
        "prompt_original_chars": len(prompt_original),
        "prompt_enviado_chars": len(prompt_ajustado or ""),
        "system_original_chars": len(system_original),
        "system_enviado_chars": len(system_ajustado or ""),
        "payload_bytes": len(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
        "recortado": (prompt_ajustado != prompt_original) or (system_ajustado != system_original),
    }

    return payload, prompt_ajustado, system_ajustado, info

def llamar_groq(prompt, system_prompt="Eres un asistente experto en poesía generativa.", model=None, allow_google_fallback=True):
    """
    Llama a Groq iterando modelos (primero el principal, luego fallbacks). Sin reintentos.
    Un error en un modelo causa siguiente modelo, no reintento.
    """
    # Lista de modelos Groq a probar (primero el modelo principal, luego fallbacks gratuitos)
    modelos_groq = [model or GROQ_MODEL, *GROQ_FALLBACK_MODELS]
    # Eliminar duplicados manteniendo orden
    modelos_groq = [modelo for modelo in dict.fromkeys(modelos_groq) if modelo]
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    print("=== DEBUG (GROQ) ===")
    print(f"Longitud prompt/system: {len(prompt or '')}/{len(system_prompt or '')} chars")

    ultimo_error = None
    for modelo_groq in modelos_groq:
        try:
            prompt_limitado, system_limitado, limite_info = _ajustar_prompt_a_limite_modelo(
                prompt,
                system_prompt,
                modelo_groq,
                "Groq",
                max_tokens_salida=1600,
            )
            payload, prompt_ajustado, system_ajustado, payload_info = _preparar_payload_groq(
                modelo_groq,
                prompt_limitado,
                system_limitado,
            )
            
            print(f"Intentando con modelo Groq: {modelo_groq}")
            if limite_info.get("recortado_por_limite_modelo"):
                print(
                    "Ajuste por límite del modelo (Groq): "
                    f"prompt={limite_info['prompt_enviado_chars']}/{limite_info['prompt_original_chars']} chars, "
                    f"system={limite_info['system_enviado_chars']}/{limite_info['system_original_chars']} chars"
                )
            if prompt_ajustado != prompt or system_ajustado != system_prompt:
                print(
                    "Payload ajustado para Groq: "
                    f"system={len(system_ajustado or '')}/{len(system_prompt or '')} chars, "
                    f"prompt={len(prompt_ajustado or '')}/{len(prompt or '')} chars"
                )
                GROQ_CALL_HISTORY.append(payload_info)
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            print(f"✓ Éxito con Groq ({modelo_groq})")
            _registrar_intento_modelo("Groq", modelo_groq, "ok", fase="chat")
            _registrar_llamada_llm("Groq", modelo_groq, solicitado=model or GROQ_MODEL)
            return _strip_think_tags(data["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"✗ Fallo con modelo Groq {modelo_groq}: {e}")
            _registrar_intento_modelo("Groq", modelo_groq, "error", error=e, fase="chat")
            ultimo_error = e
            continue

    # Fallback a Google si todos los modelos de Groq fallan
    print(f"Todos los modelos de Groq fallaron. Último error: {ultimo_error}")
    def _fallback_gemini_fast(origen_error):
        if not allow_google_fallback:
            _registrar_intento_modelo("Google", GOOGLE_FAST_MODEL, "error", error=origen_error, fase="fallback_deshabilitado")
            raise Exception(f"Error llamando a Groq sin fallback a Google permitido: {origen_error}")
        if not GOOGLE_API_KEY:
            _registrar_intento_modelo("Google", GOOGLE_FAST_MODEL, "error", error="Google API Key no configurada", fase="fallback")
            raise Exception(
                f"Error llamando a Groq y sin fallback disponible (Google API Key no configurada): {origen_error}"
            )
        if genai_module is None:
            _registrar_intento_modelo("Google", GOOGLE_FAST_MODEL, "error", error="cliente Google no inicializado", fase="fallback")
            raise Exception(
                f"Error llamando a Groq y sin fallback disponible (cliente Google no inicializado): {origen_error}"
            )
        print("Fallback activado: Groq (todos) -> Gemini Fast")
        return llamar_google(
            prompt,
            system_prompt=system_prompt,
            model=GOOGLE_FAST_MODEL,
            allow_groq_fallback=False,
        )

    return _fallback_gemini_fast(str(ultimo_error))

# ============================
#  LLAMADA A GOOGLE (TEXTO)
# ============================

def llamar_google(prompt, system_prompt=None, model=None, allow_groq_fallback=True):
    if not GOOGLE_API_KEY:
        raise Exception("Google API Key no configurada")

    if genai_module is None:
        raise Exception("Cliente de Google no inicializado")

    model_name = model or GOOGLE_MODEL
    prompt_ajustado, system_ajustado, limite_info = _ajustar_prompt_a_limite_modelo(
        prompt,
        system_prompt,
        model_name,
        "Google",
        max_tokens_salida=1600,
    )
    print("=== DEBUG (GOOGLE AI STUDIO) ===")
    print("MODEL:", model_name)
    print(f"Longitud prompt/system: {len(prompt_ajustado or '')}/{len(system_ajustado or '')} chars")
    if limite_info.get("recortado_por_limite_modelo"):
        print(
            "Ajuste por límite del modelo (Google): "
            f"prompt={limite_info['prompt_enviado_chars']}/{limite_info['prompt_original_chars']} chars, "
            f"system={limite_info['system_enviado_chars']}/{limite_info['system_original_chars']} chars"
        )

    try:
        final_prompt = prompt_ajustado
        if system_ajustado:
            final_prompt = f"INSTRUCCIONES DEL SISTEMA:\n{system_ajustado}\n\n---\n\n{prompt_ajustado}"

        model_instance = genai_module.GenerativeModel(model_name)
        response = model_instance.generate_content(final_prompt)

        _registrar_intento_modelo("Google", model_name, "ok", fase="text")
        _registrar_llamada_llm("Google", model_name)
        return _strip_think_tags(response.text)

    except Exception as e:
        print(f"Error llamando a Google AI Studio con {model_name}: {e}")
        _registrar_intento_modelo("Google", model_name, "error", error=e, fase="text")
        # Fallback a Groq si Google falla
        if GROQ_API_KEY and allow_groq_fallback:
            print("Fallback activado: Google -> Groq")
            return llamar_groq(prompt_ajustado, system_prompt=system_ajustado, allow_google_fallback=False)
        raise Exception(f"Error llamando a Google AI Studio con {model_name}: {e}")

def llamar_google_con_fallback(prompt, system_prompt=None, modelos=None):
    if not modelos:
        respuesta = llamar_google(prompt, system_prompt)
        ultima_llamada = get_last_llm_call()
        return respuesta, ((ultima_llamada or {}).get("label") or GOOGLE_MODEL)

    ultimo_error = None
    for modelo in modelos:
        try:
            respuesta = llamar_google(prompt, system_prompt, model=modelo, allow_groq_fallback=False)
            ultima_llamada = get_last_llm_call()
            return respuesta, ((ultima_llamada or {}).get("label") or modelo)
        except Exception as e:
            print(f"Fallo con el modelo Google {modelo}: {e}")
            _registrar_intento_modelo("Google", modelo, "error", error=e, fase="text")
            ultimo_error = e
            continue
            
    if GROQ_API_KEY:
        print(f"Todos los modelos de Google fallaron. Intentando fallback a Groq...")
        try:
            respuesta = llamar_groq(prompt, system_prompt=system_prompt, allow_google_fallback=False)
            ultima_llamada = get_last_llm_call()
            return respuesta, ((ultima_llamada or {}).get("label") or "Groq Fallback")
        except Exception as e:
            print(f"Todos los modelos de Groq fallaron: {e}. Intentando fallback a DeepSeek...")
            ultimo_error = e
            
    if GROQ_API_KEY:
        print(f"Google falló. Intentando fallback a Groq...")
        try:
            respuesta = llamar_groq(prompt, system_prompt=system_prompt, allow_google_fallback=False)
            ultima_llamada = get_last_llm_call()
            return respuesta, ((ultima_llamada or {}).get("label") or "Groq Fallback")
        except Exception as e:
            print(f"Groq también falló: {e}. Intentando fallback a DeepSeek...")
            ultimo_error = e

    if DEEPSEEK_API_KEY:
        try:
            respuesta = llamar_deepseek(prompt, system_prompt=system_prompt)
            ultima_llamada = get_last_llm_call()
            return respuesta, ((ultima_llamada or {}).get("label") or "DeepSeek Fallback")
        except Exception as e:
            print(f"DeepSeek también falló: {e}")
            ultimo_error = e

    raise Exception(f"Todos los modelos de fallback (Google, Groq, DeepSeek) fallaron. Último error: {ultimo_error}")

# ============================
#  LLAMADA A DEEPSEEK
# ============================

def llamar_deepseek(prompt, system_prompt="Eres un asistente experto en poesía generativa.", model=None):
    model_name = model or DEEPSEEK_MODEL
    prompt_ajustado, system_ajustado, limite_info = _ajustar_prompt_a_limite_modelo(
        prompt,
        system_prompt,
        model_name,
        "DeepSeek",
        max_tokens_salida=1600,
    )
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": model_name,
        "messages": _construir_mensajes_chat(prompt_ajustado, system_ajustado),
        "temperature": 0.9,
        "max_tokens": 1600
    }
    
    print("=== DEBUG (DEEPSEEK) ===")
    print("MODEL:", payload["model"])
    print("API KEY:", "OK" if DEEPSEEK_API_KEY else "MISSING")
    if limite_info.get("recortado_por_limite_modelo"):
        print(
            "Ajuste por límite del modelo (DeepSeek): "
            f"prompt={limite_info['prompt_enviado_chars']}/{limite_info['prompt_original_chars']} chars, "
            f"system={limite_info['system_enviado_chars']}/{limite_info['system_original_chars']} chars"
        )
    
    for intento in range(REWORK_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 429:
                print("Rate limit alcanzado. Esperando 10 segundos antes de reintentar...")
                time.sleep(10)
                continue
            response.raise_for_status()
            data = response.json()
            _registrar_intento_modelo("DeepSeek", payload["model"], "ok", fase="chat")
            _registrar_llamada_llm("DeepSeek", payload["model"])
            return _strip_think_tags(data["choices"][0]["message"]["content"])
        except Exception as e:
            if intento == REWORK_RETRIES - 1:
                _registrar_intento_modelo("DeepSeek", payload["model"], "error", error=e, fase="chat")
                # Fallback a Groq si Deepseek falla
                if GROQ_API_KEY:
                    print(f"Deepseek falló tras {REWORK_RETRIES} intentos. Fallback activado: Deepseek -> Groq")
                    return llamar_groq(prompt_ajustado, system_prompt=system_ajustado)
                raise Exception(f"Error llamando a Deepseek: {e}")
    
    raise Exception("Demasiados intentos fallidos en Deepseek")

# ============================
#  LLAMADA A GOOGLE (IMAGEN)
# ============================

def generate_from_poem(poem_text):
    """
    Uses a Gemini model to visualize a poem.
    """
    if not GOOGLE_API_KEY:
        raise Exception("Google API Key no configurada")
    if genai_module is None:
        raise Exception("Cliente de Google no inicializado")
    try:
        prompt = f'Create a highly artistic, atmospheric, and detailed visual representation inspired by this poem: "{poem_text}". Focus on the emotional resonance and symbolism.'

        # CAMBIO: Usamos un modelo de tu lista que es específico para imágenes.
        model_name = 'models/gemini-2.5-flash-image'
        print(f"--- DEBUG (GOOGLE IMAGE) ---\nMODEL: {model_name}\n--------------------------")
        
        model_instance = genai_module.GenerativeModel(model_name)
        response = model_instance.generate_content(prompt)
        
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                return part.inline_data.data
            
        return None

    except Exception as e:
        raise Exception(f"Error llamando a Google Image API: {e}")

def guardar_imagen(base64_data, ruta="imagen.png"):
    with open(ruta, "wb") as f:
        f.write(base64.b64decode(base64_data))
    return ruta

# ============================
#  LLAMADA A GOOGLE (AUDIO)
# ============================

def _limpiar_estilo_musical(texto):
    estilo = (texto or "").strip()
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

CONTEXTO POÉTICO:
{contexto_poetico}

LETRA (poema):
{poema_texto}
"""
    # CAMBIO: Usamos los modelos PRO que tienes disponibles en tu lista.
    modelos_a_probar = [model] if model else ["gemini-pro-latest", "models/gemini-2.5-pro"]
    estilo, modelo_usado = llamar_google_con_fallback(prompt, modelos=modelos_a_probar)
    return _limpiar_estilo_musical(estilo), modelo_usado

def _generar_audio_con_google(prompt_audio, model=None, voice_name=None):
    if genai_module is None:
        raise Exception("Cliente de Google no inicializado")

    # CORRECCIÓN: La ruta correcta es 'google.generativeai.types'
    types_module = importlib.import_module("google.generativeai.types")
    selected_model = model or GOOGLE_MUSIC_MODEL
    es_lyria = "lyria" in selected_model.lower()

    print(f"--- DEBUG (GOOGLE AUDIO) ---\nMODEL: {selected_model}\n--------------------------")

    if es_lyria:
        config = types_module.GenerateContentConfig(
            response_modalities=["AUDIO"],
        )
    else:
        config = types_module.GenerationConfig(
            text_to_speech_config=types_module.TextToSpeechConfig(
                speech_config=types_module.SpeechConfig(
                    voice_config=types_module.VoiceConfig(
                        prebuilt_voice_config=types_module.PrebuiltVoiceConfig(
                            voice_name=voice_name or GOOGLE_MUSIC_VOICE
                        )
                    )
                )
            ),
        )

    model_instance = genai_module.GenerativeModel(selected_model)
    response = model_instance.generate_content(
        contents=prompt_audio,
        generation_config=config,
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
    if estilo_musical_override:
        estilo_musical = _limpiar_estilo_musical(estilo_musical_override)
    else:
        estilo_musical, _ = generar_estilo_musical_desde_contexto(
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
        "model_used": GOOGLE_MUSIC_MODEL,
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
    return random.sample(lista, min(k, len(lista)))
