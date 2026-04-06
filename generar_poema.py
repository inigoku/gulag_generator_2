# agente.py

import json

from clasificar_intencion_poetica import clasificar_intencion_poetica
from chroma import abrir_o_reconstruir_chroma, buscar_en_chroma
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
    from utils_llamadas import generate_from_poem
    return generate_from_poem(poema)


def _parsear_critica_json(critica_raw):
    if isinstance(critica_raw, dict):
        return critica_raw

    if isinstance(critica_raw, str):
        try:
            start = critica_raw.find("{")
            end = critica_raw.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(critica_raw[start:end])
        except Exception:
            pass

    return {"ok": False, "problemas": ["Error formato JSON"], "sugerencias": []}


def ejecutar_rework_poetico(params):
    base = "."
    loguear_etapa = params.get("loguear_etapa")

    poema_base = (params.get("poema_base") or "").strip()
    comentario_rework = (params.get("comentario_rework") or "").strip()
    if not poema_base:
        raise Exception("No hay poema base para afinar.")

    perfil_estilistico = leer_texto(f"{base}/estilo/perfil_estilistico_final.md")
    prompt_eval = cargar_prompt(f"{base}/prompts/prompt_evaluacion.txt")
    prompt_rewrite = cargar_prompt(f"{base}/prompts/prompt_reescritura.txt")
    prompt_pulido = cargar_prompt(f"{base}/prompts/prompt_pulido_final.txt")

    groq_model = params.get("groq_model")
    google_model = params.get("google_model")
    tema = params.get("tema", "")
    tono_extra = params.get("tono_extra", "")
    contexto_extendido = params.get("contexto_extendido", "")

    # Si no hay contexto extenso (p.ej., guardados antiguos), construimos uno mínimo
    # para no romper la etapa de pulido final.
    if not (contexto_extendido or "").strip():
        contexto_extendido = (
            "PERFIL_ESTILISTICO:\n"
            f"{perfil_estilistico}\n\n"
            "CONTEXTO_MINIMO_REWORK:\n"
            f"TEMA: {tema}\n"
            f"TONO_EXTRA: {tono_extra}\n"
            "ORIGEN: rework sobre poema ya generado"
        )

    if loguear_etapa:
        loguear_etapa("Afinado - Entrada", f"Tema: {tema}", f"Comentario: {comentario_rework}\n\nPoema base:\n{poema_base}")

    # 1) REWORK inicial guiado por comentario del usuario
    prompt_rework = (
        f"{prompt_rewrite}\n\n"
        f"COMENTARIO DEL USUARIO (prioritario):\n{comentario_rework or 'Sin comentario explícito.'}\n\n"
        f"POEMA ORIGINAL:\n{poema_base}\n\n"
        f"PROBLEMAS:\nDerivados del comentario del usuario\n\n"
        f"SUGERENCIAS:\nAplica exactamente la intención del comentario manteniendo coherencia y calidad poética\n\n"
        f"ESTILO:\n{perfil_estilistico}"
    )
    POEMA_CORREGIDO = llamar_groq(prompt_rework, system_prompt="Eres un editor de poesía experto.", model=groq_model)
    if loguear_etapa:
        loguear_etapa("Afinado - Rework inicial", prompt_rework, POEMA_CORREGIDO)

    # 2) EVALUACIÓN + 3) REESCRITURA iterativa
    prompt_evaluacion = f"{prompt_eval}\n\nPOEMA:\n{POEMA_CORREGIDO}\n\nESTILO:\n{perfil_estilistico}\n\nTEMA:\n{tema}"
    CRITICA = _parsear_critica_json(
        llamar_groq(
            prompt_evaluacion,
            system_prompt="Eres un crítico literario experto. Responde estrictamente en JSON.",
            model=groq_model,
        )
    )
    if loguear_etapa:
        loguear_etapa("Afinado - Evaluación", prompt_evaluacion, str(CRITICA))

    iteraciones = 0
    max_iter = 3
    while not CRITICA.get("ok", False) and iteraciones < max_iter:
        probs = ", ".join(CRITICA.get("problemas", []))
        sugs = ", ".join(CRITICA.get("sugerencias", []))
        prompt_reescritura = (
            f"{prompt_rewrite}\n\n"
            f"COMENTARIO DEL USUARIO (mantener intención):\n{comentario_rework or 'Sin comentario explícito.'}\n\n"
            f"POEMA ORIGINAL:\n{POEMA_CORREGIDO}\n\n"
            f"PROBLEMAS:\n{probs}\n\n"
            f"SUGERENCIAS:\n{sugs}\n\n"
            f"ESTILO:\n{perfil_estilistico}"
        )
        POEMA_CORREGIDO = llamar_groq(prompt_reescritura, system_prompt="Eres un editor de poesía experto.", model=groq_model)
        if loguear_etapa:
            loguear_etapa("Afinado - Reescritura", prompt_reescritura, POEMA_CORREGIDO)

        prompt_evaluacion = f"{prompt_eval}\n\nPOEMA:\n{POEMA_CORREGIDO}\n\nESTILO:\n{perfil_estilistico}\n\nTEMA:\n{tema}"
        CRITICA = _parsear_critica_json(
            llamar_groq(
                prompt_evaluacion,
                system_prompt="Eres un crítico literario experto. Responde estrictamente en JSON.",
                model=groq_model,
            )
        )
        if loguear_etapa:
            loguear_etapa("Afinado - Reevaluación", prompt_evaluacion, str(CRITICA))

        iteraciones += 1

    # 4) PULIDO FINAL
    prompt_pulido_final = f"{contexto_extendido}\n\nPOEMA PREVIO:\n{POEMA_CORREGIDO}\n\nINSTRUCCIONES DE PULIDO:\n{prompt_pulido}"
    POEMA_FINAL = llamar_google(prompt_pulido_final, model=google_model)
    if loguear_etapa:
        loguear_etapa("Afinado - Pulido final", prompt_pulido_final, POEMA_FINAL)

    return {
        "poema_final": POEMA_FINAL,
        "poema_inicial": poema_base,
        "poema_corregido": POEMA_CORREGIDO,
        "critica_final": CRITICA,
        "contexto_extendido": contexto_extendido,
        "comentario_rework": comentario_rework,
        "insumos_pulido": {
            "contexto_extendido": contexto_extendido,
            "tema": tema,
            "tono_extra": tono_extra,
        },
    }

def ejecutar_pipeline_poetico(params):
    base = "."
    loguear_etapa = params.get("loguear_etapa")

    # 0. RECUPERAR DATOS
    perfil_estilistico = leer_texto(f"{base}/estilo/perfil_estilistico_final.md")
    if loguear_etapa:
        loguear_etapa("Recuperar perfil estilístico", f"Ruta: {base}/estilo/perfil_estilistico_final.md", perfil_estilistico)

    chunks_obra = cargar_json(f"{base}/data/chunks/chunks_obra.json")
    chunks_influencias = cargar_json(f"{base}/data/chunks/chunks_influencias.json")
    chunks_folklore = cargar_json(f"{base}/data/chunks/chunks_folklore.json")
    if loguear_etapa:
        loguear_etapa("Cargar chunks obra", f"Ruta: {base}/data/chunks/chunks_obra.json", str(chunks_obra))
        loguear_etapa("Cargar chunks influencias", f"Ruta: {base}/data/chunks/chunks_influencias.json", str(chunks_influencias))
        loguear_etapa("Cargar chunks folklore", f"Ruta: {base}/data/chunks/chunks_folklore.json", str(chunks_folklore))

    prompt_maestro = cargar_prompt(f"{base}/prompts/prompt_maestro.txt")
    prompt_eval = cargar_prompt(f"{base}/prompts/prompt_evaluacion.txt")
    prompt_rewrite = cargar_prompt(f"{base}/prompts/prompt_reescritura.txt")
    prompt_pulido = cargar_prompt(f"{base}/prompts/prompt_pulido_final.txt")
    if loguear_etapa:
        loguear_etapa("Cargar prompts", "Prompts cargados", f"Maestro: {prompt_maestro}\nEvaluación: {prompt_eval}\nReescritura: {prompt_rewrite}\nPulido: {prompt_pulido}")

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
    if loguear_etapa:
        loguear_etapa("Clasificación de intención poética", f"Tema: {params.get('tema', '')}", str(perfil))

    # 2. NUEVO: GENERAR ESTRUCTURA POÉTICA
    estructura = generar_estructura_poetica(perfil)
    if not isinstance(estructura, dict):
        estructura = {}
    perfil["estructura"] = estructura
    if loguear_etapa:
        loguear_etapa("Generar estructura poética", "Perfil", str(estructura))

    # 3. SISTEMA DE PESOS ADAPTATIVO
    pesos = calcular_pesos(perfil)
    if loguear_etapa:
        loguear_etapa("Calcular pesos", "Perfil", str(pesos))
    

    # 3.1. ACTIVAR BRAVE SEARCH SEGÚN γ
    contexto_factual = ""
    if pesos["γ"] > 0.15:
        resultados = brave_search(params.get("tema", ""))
        contexto_factual = "\n\n".join(resultados[:5])
        if loguear_etapa:
            loguear_etapa("Brave Search", f"Tema: {params.get('tema', '')}", contexto_factual)
        
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
    if loguear_etapa:
        loguear_etapa("Rigidez estructural", "Perfil", str(rigidez))

    # 6. CONSTRUIR CONTEXTO LARGO (GEMINI) — VERSIÓN SEGURA
    chroma_obra = abrir_o_reconstruir_chroma(f"{base}/data/chroma/obra/", chunks_obra)
    chroma_influencias = abrir_o_reconstruir_chroma(f"{base}/data/chroma/influencias/", chunks_influencias)
    chroma_folklore = abrir_o_reconstruir_chroma(f"{base}/data/chroma/folklore/", chunks_folklore)
    fragmentos_obra = buscar_en_chroma(chroma_obra, params['tema'], k=30)
    fragmentos_influencias = buscar_en_chroma(chroma_influencias, params['tema'], k=30)
    fragmentos_folklore = buscar_en_chroma(chroma_folklore, params['tema'], k=30)
    if loguear_etapa:
        loguear_etapa("Buscar en Chroma obra", f"Tema: {params['tema']}", str(fragmentos_obra))
        loguear_etapa("Buscar en Chroma influencias", f"Tema: {params['tema']}", str(fragmentos_influencias))
        loguear_etapa("Buscar en Chroma folklore", f"Tema: {params['tema']}", str(fragmentos_folklore))

    texto_obra = "\n\n".join(fragmentos_obra)
    texto_influencias = "\n\n".join(fragmentos_influencias)
    texto_folklore = "\n\n".join(fragmentos_folklore)

    instrucciones_formateadas = prompt_maestro.format(
        estilo=perfil_estilistico,
        estructura=EstructuraFlexible(estructura),
        mezcla=texto_obra,
        influencias=texto_influencias,
        folklore=texto_folklore,
        tema=params.get("tema", ""),
        tono_extra=params.get("tono_extra", ""),
        restricciones=params.get("restricciones", ""),
        extension=params.get("extension", "")
    )
    if loguear_etapa:
        loguear_etapa("Formatear instrucciones", "Prompt maestro", instrucciones_formateadas)

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

    CONTEXTO_FOLKLORE:
    {texto_folklore}

    CONTEXTO_FACTUAL (solo si γ > 0.15):
    {contexto_factual}

    INSTRUCCIONES:
    {instrucciones_formateadas}
    """
    if loguear_etapa:
        loguear_etapa("Construir contexto extendido", "Contexto", CONTEXTO_EXTENDIDO)

    # 7. GENERACIÓN
    prompt_generacion = f"{CONTEXTO_EXTENDIDO}\n\nTAREA:\nEscribe un poema sobre: {params['tema']}"
    POEMA_INICIAL = llamar_google(prompt_generacion, model=google_model)
    if loguear_etapa:
        loguear_etapa("Generar poema inicial", prompt_generacion, POEMA_INICIAL)

    # 8. EVALUACIÓN
    prompt_evaluacion = f"{prompt_eval}\n\nPOEMA:\n{POEMA_INICIAL}\n\nESTILO:\n{perfil_estilistico}\n\nTEMA:\n{params['tema']}"
    CRITICA = llamar_groq(prompt_evaluacion, system_prompt="Eres un crítico literario experto. Responde estrictamente en JSON.", model=groq_model)
    if loguear_etapa:
        loguear_etapa("Evaluar poema inicial", prompt_evaluacion, str(CRITICA))

    # 9. REESCRITURA
    POEMA_CORREGIDO = POEMA_INICIAL
    iteraciones = 0
    max_iter = 3

    while not (isinstance(CRITICA, dict) and CRITICA.get("ok", False)) and iteraciones < max_iter:
        probs = ", ".join(CRITICA.get("problemas", [])) if isinstance(CRITICA, dict) else ""
        sugs = ", ".join(CRITICA.get("sugerencias", [])) if isinstance(CRITICA, dict) else ""
        prompt_reescritura = f"{prompt_rewrite}\n\nPOEMA ORIGINAL:\n{POEMA_CORREGIDO}\n\nPROBLEMAS:\n{probs}\n\nSUGERENCIAS:\n{sugs}\n\nESTILO:\n{perfil_estilistico}"
        POEMA_CORREGIDO = llamar_groq(prompt_reescritura, system_prompt="Eres un editor de poesía experto.", model=groq_model)
        if loguear_etapa:
            loguear_etapa("Reescribir poema", prompt_reescritura, POEMA_CORREGIDO)
        prompt_evaluacion_corregido = f"{prompt_eval}\n\nPOEMA:\n{POEMA_CORREGIDO}\n\nESTILO:\n{perfil_estilistico}\n\nTEMA:\n{params['tema']}"
        CRITICA = llamar_groq(prompt_evaluacion_corregido, system_prompt="Eres un crítico literario experto. Responde estrictamente en JSON.", model=groq_model)
        if loguear_etapa:
            loguear_etapa("Evaluar poema corregido", prompt_evaluacion_corregido, str(CRITICA))
        iteraciones += 1

    # 10. PULIDO FINAL
    prompt_pulido_final = f"{CONTEXTO_EXTENDIDO}\n\nPOEMA PREVIO:\n{POEMA_CORREGIDO}\n\nINSTRUCCIONES DE PULIDO:\n{prompt_pulido}"
    POEMA_FINAL = llamar_google(prompt_pulido_final, model=google_model)
    if loguear_etapa:
        loguear_etapa("Pulido final", prompt_pulido_final, POEMA_FINAL)

    # 11. GENERAR IMAGEN (Opcional)
    imagen = None
    if params.get("crear_imagen"):
        try:
            from utils_llamadas import generate_from_poem
            imagen = generate_from_poem(POEMA_FINAL)
            if loguear_etapa:
                loguear_etapa("Generar imagen", "Poema final", str(imagen))
        except Exception as e:
            if loguear_etapa:
                loguear_etapa("Error generando imagen", "Poema final", str(e))
            imagen = None

    # 12. GENERAR AUDIO (Opcional)
    audio_url = None
    audio_bytes = None
    audio_estilo = None
    audio_task_id = None
    audio_status = None
    audio_error = None
    if params.get("crear_audio"):
        try:
            from utils_llamadas import generar_audio_poema_con_suno
            audio_resultado = generar_audio_poema_con_suno(
                poema_texto=POEMA_FINAL,
                contexto_poetico=CONTEXTO_EXTENDIDO,
                tema=params.get("tema", ""),
                tono_extra=params.get("tono_extra", ""),
                model_google=google_model,
                titulo=params.get("audio_title", ""),
                instrumental=params.get("audio_instrumental", False),
                wait_audio=params.get("audio_wait", True),
                duracion_segundos=params.get("audio_duracion_seg", 90)
            )
            audio_url = audio_resultado.get("audio_url")
            audio_bytes = audio_resultado.get("audio_bytes")
            audio_estilo = audio_resultado.get("estilo_musical")
            audio_task_id = audio_resultado.get("task_id")
            audio_status = audio_resultado.get("status")
            if loguear_etapa:
                loguear_etapa("Generar audio", "Poema final + contexto", str(audio_resultado))
        except Exception as e:
            audio_error = str(e)
            if loguear_etapa:
                loguear_etapa("Error generando audio", "Poema final + contexto", audio_error)

    return {
        "poema_final": POEMA_FINAL,
        "poema_inicial": POEMA_INICIAL,
        "poema_corregido": POEMA_CORREGIDO,
        "critica_final": CRITICA,
        "contexto_extendido": CONTEXTO_EXTENDIDO,
        "insumos_pulido": {
            "contexto_extendido": CONTEXTO_EXTENDIDO,
            "tema": params.get("tema", ""),
            "tono_extra": params.get("tono_extra", ""),
        },
        "historial_afinados": [],
        "estructura": estructura,
        "pesos": pesos,
        "perfil": perfil,
        "imagen": imagen,
        "audio_url": audio_url,
        "audio_bytes": audio_bytes,
        "audio_estilo": audio_estilo,
        "audio_task_id": audio_task_id,
        "audio_status": audio_status,
        "audio_error": audio_error
    }