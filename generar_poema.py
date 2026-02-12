# agente.py

import json

from clasificar_intencion_poetica import clasificar_intencion_poetica
from generar_estructura_poetica import generar_estructura_poetica
from calcular_pesos import calcular_pesos
from brave_search import brave_search
from utils_llamadas import llamar_groq, llamar_google, cargar_prompt, leer_texto, cargar_json, seleccionar

class EstructuraFlexible(dict):
    """Permite acceso por punto (para prompt) y por clave (para f-strings)"""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            return f"[{key} no definido]"

###############################################################################
# NUEVO PIPELINE POÉTICO (Basado en archivos)
###############################################################################


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

def generar_imagen(poema, model=None):
    return llamar_imagen(poema, model=model)

def ejecutar_pipeline_poetico(params):
    base = "."
    
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
    crear_imagen = params.get("crear_imagen", False)

    # 1. CLASIFICACIÓN DE INTENCIÓN POÉTICA
    perfil = clasificar_intencion_poetica(
        params.get("tema", ""),
        params.get("estilo_extra", ""),
        params.get("tono_extra", ""),
        params.get("restricciones", ""),
        params.get("extension", "")
    )

    # 2. NUEVO: GENERAR ESTRUCTURA POÉTICA
    estructura = generar_estructura_poetica(perfil)
    if not isinstance(estructura, dict):
        estructura = {}
    perfil["estructura"] = estructura

    # 3. SISTEMA DE PESOS ADAPTATIVO
    pesos = calcular_pesos(perfil)
    

    # 3.1. ACTIVAR BRAVE SEARCH SEGÚN γ
    contexto_factual = ""
    if pesos["γ"] > 0.15:
        resultados = brave_search(params.get("tema", ""))
        contexto_factual = "\n\n".join(resultados[:5])
        
    # 3.2. FLEXIBILIDAD ESTRUCTURAL SEGÚN INTENCIÓN
    rigidez = pesos.get("rigidez_estructural", 0.5)

    # Si la intención es más libre, reducimos rigidez
    if perfil.get("intencion") in ["lírica", "fluida", "experimental"]:
        rigidez *= 0.5

    # Si el usuario pide libertad explícita
    if "libre" in params.get("restricciones", "").lower():
        rigidez = 0.0

    # Guardamos la rigidez para el prompt
    perfil["rigidez"] = rigidez

    # 6. CONSTRUIR CONTEXTO LARGO (GEMINI) — VERSIÓN SEGURA
    # Seleccionar solo los fragmentos necesarios
    fragmentos_obra = seleccionar(chunks_obra, k=8)
    fragmentos_influencias = seleccionar(chunks_influencias, k=5)

    # Compactar los fragmentos seleccionados
    texto_obra = "\n\n".join(fragmentos_obra)
    texto_influencias = "\n\n".join(fragmentos_influencias)

    # Formatear las instrucciones primero para evitar conflictos con llaves en los textos
    instrucciones_formateadas = prompt_maestro.format(
        estilo=perfil_estilistico,
        estructura=EstructuraFlexible(estructura),
        mezcla=texto_obra,
        influencias=texto_influencias,
        tema=params.get("tema", ""),
        tono_extra=params.get("tono_extra", ""),
        restricciones=params.get("restricciones", ""),
        extension=params.get("extension", "")
    )


    # Construir el contexto extendido compacto
    CONTEXTO_EXTENDIDO = f"""
    PERFIL_ESTILISTICO:
    {perfil_estilistico}

    ESTRUCTURA_POETICA:
    Número de estrofas: {estructura['num_estrofas']}
    Versos por estrofa: {estructura['versos_por_estrofa']}
    Tipo de verso: {estructura['tipo_verso']}
    Ritmo: {estructura['ritmo']}
    Progresión: {estructura['progresion']}
    Notas: {estructura['notas']}

    CONTEXTO_OBRA:
    {texto_obra}

    CONTEXTO_INFLUENCIAS:
    {texto_influencias}

    CONTEXTO_FACTUAL (solo si γ > 0.15):
    {contexto_factual}

    INSTRUCCIONES:
    {instrucciones_formateadas}
    """

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

    # 11. GENERAR IMAGEN (Opcional)
    imagen = None

    if params.get("crear_imagen"):
        try:
            from utils_llamadas import generate_from_poem
            imagen = generate_from_poem(POEMA_FINAL)
        except Exception as e:
            print("Error generando imagen:", e)
            imagen = None


    return {
        "poema_final": POEMA_FINAL,
        "poema_inicial": POEMA_INICIAL,
        "poema_corregido": POEMA_CORREGIDO,
        "critica_final": CRITICA,
        "estructura": estructura,
        "pesos": pesos,
        "perfil": perfil,
        "imagen": imagen
    }