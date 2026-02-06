# agente.py

import json
import requests
import itertools
import random
import os

from config import (
    GROQ_API_KEY, GROQ_MODEL, REWORK_RETRIES,
    GOOGLE_MODEL, GOOGLE_API_KEY
)

import unicodedata

def limpiar_prompt(texto):
    # Normaliza unicode (NFC) pero mantiene caracteres latinos como tildes y ñ
    texto = unicodedata.normalize("NFC", texto)
    return texto.strip()

# Configuración para Google AI Studio (Capa gratuita sin Vertex)
if GOOGLE_API_KEY:
    from google import genai
    client = genai.Client(api_key=GOOGLE_API_KEY)

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
    max_retries = 5
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

###############################################################################
# NUEVO PIPELINE POÉTICO (Basado en archivos)
###############################################################################

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

def gemini_generar_poema(contexto, user, model=None):
    prompt = f"{contexto}\n\nTAREA:\n{user}"
    return llamar_google(prompt, model=model)

def groq_evaluar_poema(poema, prompt, estilo, tema, model=None):
    full_prompt = f"{prompt}\n\nPOEMA:\n{poema}\n\nESTILO:\n{estilo}\n\nTEMA:\n{tema}"
    resp = llamar_groq(full_prompt, system_prompt="Eres un crítico literario experto. Responde estrictamente en JSON.", model=model)
    try:
        start = resp.find('{')
        end = resp.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(resp[start:end])
    except:
        pass
    return {"ok": False, "problemas": ["Error formato JSON"], "sugerencias": []}

def groq_reescribir_poema(poema, prompt, problemas, sugerencias, estilo, model=None):
    probs = ", ".join(problemas)
    sugs = ", ".join(sugerencias)
    full_prompt = f"{prompt}\n\nPOEMA ORIGINAL:\n{poema}\n\nPROBLEMAS:\n{probs}\n\nSUGERENCIAS:\n{sugs}\n\nESTILO:\n{estilo}"
    return llamar_groq(full_prompt, system_prompt="Eres un editor de poesía experto.", model=model)

def gemini_pulir_poema(contexto, poema, prompt, model=None):
    full_prompt = f"{contexto}\n\nPOEMA PREVIO:\n{poema}\n\nINSTRUCCIONES DE PULIDO:\n{prompt}"
    return llamar_google(full_prompt, model=model)

def ejecutar_pipeline_poetico(params):
    base = "./generador_v2"
    
    # 0. RECUPERAR DATOS
    perfil_estilistico = leer_texto(f"{base}/estilo/perfil_estilistico_final.md")
    
    chunks_obra = cargar_json(f"{base}/data/chunks/chunks_obra.json")
    chunks_influencias = cargar_json(f"{base}/data/chunks/chunks_influencias.json")
    
    prompt_maestro = cargar_prompt(f"{base}/prompts/prompt_maestro.txt")
    prompt_eval = cargar_prompt(f"{base}/prompts/prompt_evaluacion.txt")
    prompt_rewrite = cargar_prompt(f"{base}/prompts/prompt_reescritura.txt")
    prompt_pulido = cargar_prompt(f"{base}/prompts/prompt_pulido_final.txt")
    
    groq_model = params.get("groq_model")
    google_model = params.get("google_model")

    # 6. CONSTRUIR CONTEXTO LARGO (GEMINI) — VERSIÓN SEGURA
    # Seleccionar solo los fragmentos necesarios
    fragmentos_obra = seleccionar(chunks_obra, k=8)
    fragmentos_influencias = seleccionar(chunks_influencias, k=5)

    # Compactar los fragmentos seleccionados
    texto_obra = "\n\n".join(fragmentos_obra)
    texto_influencias = "\n\n".join(fragmentos_influencias)

    # Construir el contexto extendido compacto
    CONTEXTO_EXTENDIDO = f"""
PERFIL_ESTILISTICO:
{perfil_estilistico}

CONTEXTO_OBRA (fragmentos relevantes):
{texto_obra}

CONTEXTO_INFLUENCIAS (fragmentos relevantes):
{texto_influencias}

INSTRUCCIONES:
{prompt_maestro}
"""
    
    CONTEXTO_EXTENDIDO=CONTEXTO_EXTENDIDO.format(
        estilo=params.get("estilo", ""),
        mezcla=params.get("mezcla", ""),
        tema=params.get("tema", ""),
        tono_extra=params.get("tono_extra", ""),
        restricciones=params.get("restricciones", ""),
        extension=params.get("extension", 14),
)

    # 7. GENERACIÓN
    POEMA_INICIAL = gemini_generar_poema(CONTEXTO_EXTENDIDO, f"Escribe un poema sobre: {params['tema']}", model=google_model)

    # 8. EVALUACIÓN
    CRITICA = groq_evaluar_poema(POEMA_INICIAL, prompt_eval, perfil_estilistico, params['tema'], model=groq_model)

    # 9. REESCRITURA
    POEMA_CORREGIDO = POEMA_INICIAL
    iteraciones = 0
    max_iter = 3

    while not CRITICA.get("ok", False) and iteraciones < max_iter:
        POEMA_CORREGIDO = groq_reescribir_poema(
            POEMA_CORREGIDO, prompt_rewrite, 
            CRITICA.get("problemas", []), CRITICA.get("sugerencias", []), 
            perfil_estilistico, model=groq_model
        )
        CRITICA = groq_evaluar_poema(POEMA_CORREGIDO, prompt_eval, perfil_estilistico, params['tema'], model=groq_model)
        iteraciones += 1

    # 10. PULIDO FINAL
    POEMA_FINAL = gemini_pulir_poema(CONTEXTO_EXTENDIDO, POEMA_CORREGIDO, prompt_pulido, model=google_model)

    return {
        "poema_final": POEMA_FINAL,
        "poema_inicial": POEMA_INICIAL,
        "poema_corregido": POEMA_CORREGIDO,
        "critica_final": CRITICA
    }