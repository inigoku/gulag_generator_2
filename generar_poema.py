import json

from clasificar_intencion_poetica import clasificar_intencion_poetica
from chroma import abrir_chroma, buscar_en_chroma
from generar_estructura_poetica import generar_estructura_poetica
from calcular_pesos import calcular_pesos
from brave_search import brave_search
from utils_llamadas import (
    llamar_groq, llamar_google, llamar_google_con_fallback, llamar_deepseek,
    cargar_prompt, leer_texto, cargar_json, seleccionar,
    reset_groq_call_history, get_groq_call_history, get_last_llm_call,
    get_model_attempt_history,
)

# Límits de caràcters per compactar fragments de Chroma
MAX_CHARS_CONTEXTO_OBRA         = 3000
MAX_CHARS_CONTEXTO_INFLUENCIAS  = 2000
MAX_CHARS_CONTEXTO_FOLKLORE     = 1000
MAX_CHARS_CONTEXTO_EXTENDIDO    = 500
NUM_EXTENDIDOS_DEFAULT          = 10

MAX_RESULTADOS_FACTUALES = 3

MODELOS_PRO   = ["gemini-pro-latest", "gemini-3-pro-preview", "gemini-flash-latest", "gemini-flash-lite-latest"]
MODELOS_FLASH = ["gemini-flash-latest", "gemini-flash-lite-latest"]


# ---------------------------------------------------------------------------
# Utilitats internes
# ---------------------------------------------------------------------------

def _label_ultima_llamada(predeterminado):
    ultima_llamada = get_last_llm_call()
    return (ultima_llamada or {}).get("label") or predeterminado


def _compactar_fragmentos(fragmentos, max_chars):
    seleccionados = []
    chars_actuales = 0
    for fragmento in fragmentos or []:
        texto = (fragmento or "").strip()
        if not texto:
            continue
        extra = len(texto) + (2 if seleccionados else 0)
        if seleccionados and chars_actuales + extra > max_chars:
            break
        if not seleccionados and len(texto) > max_chars:
            seleccionados.append(texto[:max_chars])
            break
        seleccionados.append(texto)
        chars_actuales += extra
    return "\n\n".join(seleccionados)


class EstructuraFlexible(dict):
    """Permet accés per punt (per a prompt.format) i per clau (per a f-strings)."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            return f"[{key} no definido]"


def _parsear_critica_json(critica_raw):
    if isinstance(critica_raw, dict):
        return critica_raw
    if isinstance(critica_raw, str):
        try:
            start = critica_raw.find("{")
            end   = critica_raw.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(critica_raw[start:end])
        except Exception:
            pass
    return {"ok": False, "problemas": ["Error formato JSON"], "sugerencias": []}


# ---------------------------------------------------------------------------
# NOU: traducció de pesos a paràmetres concrets
# ---------------------------------------------------------------------------

def pesos_a_params(pesos):
    """
    Tradueix el dict de pesos normalitzats a paràmetres concrets per al pipeline:
      - k_* per a les crides a Chroma (fragments recuperats)
      - peso_* per injectar al prompt maestro com a orientació quantitativa
      - γ per al llindar de Brave Search
    """
    return {
        "k_obra":        pesos.get("k_obra", 6),
        "k_influencies": pesos.get("k_influencies", 4),
        "k_folklore":    pesos.get("k_folklore", 2),
        "k_extendido":   pesos.get("k_extendido", 1),
        "γ":             pesos.get("γ", 0.0),
        "peso_factual":  round(pesos.get("γ", 0.0), 3),
        "peso_literari": round(pesos.get("β", 0.0), 3),
        "peso_estil":    round(pesos.get("α", 0.0), 3),
    }


# ---------------------------------------------------------------------------
# Generació d'imatge
# ---------------------------------------------------------------------------

def generar_imagen(poema, model=None):
    from utils_llamadas import generate_from_poem
    return generate_from_poem(poema)


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def ejecutar_pipeline_poetico(params):
    base = "."
    loguear_etapa = params.get("loguear_etapa")
    num_extendidos = max(1, int(params.get("num_extendidos", NUM_EXTENDIDOS_DEFAULT)))
    reset_groq_call_history()

    # 0. RECUPERAR DADES
    perfil_estilistico = leer_texto(f"{base}/estilo/perfil_estilistico_final.md")
    if loguear_etapa:
        loguear_etapa("Recuperar perfil estilístico",
                      f"Ruta: {base}/estilo/perfil_estilistico_final.md",
                      perfil_estilistico)

    chunks_obra         = cargar_json(f"{base}/data/chunks/chunks_obra.json")
    chunks_influencias  = cargar_json(f"{base}/data/chunks/chunks_influencias.json")
    chunks_folklore     = cargar_json(f"{base}/data/chunks/chunks_folklore.json") or []
    chunks_extendido = sum(
        [cargar_json(f"{base}/data/chunks/chunks_extendido{i}.json") or []
         for i in range(1, num_extendidos + 1)],
        []
    )
    if loguear_etapa:
        loguear_etapa("Cargar chunks", "Tots els chunks carregats",
                      f"obra={len(chunks_obra or [])}, "
                      f"influencias={len(chunks_influencias or [])}, "
                      f"folklore={len(chunks_folklore)}, "
                      f"extendido={len(chunks_extendido)} "
                      f"(bloques={num_extendidos})")

    prompt_maestro = cargar_prompt(f"{base}/prompts/prompt_maestro.txt")
    prompt_eval    = cargar_prompt(f"{base}/prompts/prompt_evaluacion.txt")
    prompt_rewrite = cargar_prompt(f"{base}/prompts/prompt_reescritura.txt")
    prompt_pulido  = cargar_prompt(f"{base}/prompts/prompt_pulido_final.txt")

    # 1. CLASSIFICACIÓ D'INTENCIÓ POÈTICA
    perfil = clasificar_intencion_poetica(
        params.get("tema", ""),
        params.get("estilo_extra", ""),
        params.get("tono_extra", ""),
        params.get("restricciones", ""),
        params.get("extension", ""),
    )
    if loguear_etapa:
        loguear_etapa("Clasificación de intención poética",
                      f"Tema: {params.get('tema', '')}", str(perfil))

    modelos_usados_info = {
        "clasificacion_intencion": _label_ultima_llamada("DeepSeek"),
    }

    # 2. GENERAR ESTRUCTURA POÈTICA
    estructura = generar_estructura_poetica(perfil)
    if not isinstance(estructura, dict):
        estructura = {}
    perfil["estructura"] = estructura
    if loguear_etapa:
        loguear_etapa("Generar estructura poética", "Perfil", str(estructura))
    modelos_usados_info["generacion_estructura"] = _label_ultima_llamada("DeepSeek")

    # 3. SISTEMA DE PESOS ADAPTATIUS
    pesos      = calcular_pesos(perfil)
    params_ctx = pesos_a_params(pesos)   # k_* i peso_* derivats
    if loguear_etapa:
        loguear_etapa("Calcular pesos", "Perfil", str(pesos))
        loguear_etapa("Params context derivats", "pesos_a_params", str(params_ctx))

    # 3.1. BRAVE SEARCH si γ > 0.15
    contexto_factual = ""
    if params_ctx["γ"] > 0.15:
        resultados = brave_search(params.get("tema", ""))
        contexto_factual = "\n\n".join(resultados[:MAX_RESULTADOS_FACTUALES])
        if loguear_etapa:
            loguear_etapa("Brave Search",
                          f"Tema: {params.get('tema', '')}", contexto_factual)

    # 3.2. RIGIDESA ESTRUCTURAL (ara ve de calcular_pesos, no és mai 0.5 per defecte)
    rigidez = pesos.get("rigidez_estructural", 0.5)
    if perfil.get("intencion") in ["lírica", "fluida", "experimental"]:
        rigidez *= 0.5
    if "libre" in params.get("restricciones", "").lower():
        rigidez = 0.0
    perfil["rigidez"] = rigidez
    if loguear_etapa:
        loguear_etapa("Rigidez estructural", "Perfil", str(rigidez))

    # 4. OBRIR COL·LECCIONS CHROMA
    chroma_obra         = abrir_chroma(f"{base}/data/chroma/obra/")
    chroma_influencias  = abrir_chroma(f"{base}/data/chroma/influencias/")
    chroma_folklore1    = abrir_chroma(f"{base}/data/chroma/folklore1/")
    chroma_folklore2    = abrir_chroma(f"{base}/data/chroma/folklore2/")
    chromas_extendido = [
        abrir_chroma(f"{base}/data/chroma/extendido{i}/")
        for i in range(1, num_extendidos + 1)
    ]

    # 5. CERCAR FRAGMENTS — k dinàmics derivats dels pesos
    k_obra        = params_ctx["k_obra"]
    k_influencies = params_ctx["k_influencies"]
    k_folklore    = params_ctx["k_folklore"]
    k_extendido   = params_ctx["k_extendido"]
    tema          = params["tema"]

    fragmentos_obra        = buscar_en_chroma(chroma_obra,        tema, k=k_obra)
    fragmentos_influencias = buscar_en_chroma(chroma_influencias, tema, k=k_influencies)
    fragmentos_folklore1   = buscar_en_chroma(chroma_folklore1,   tema, k=k_folklore)
    fragmentos_folklore2   = buscar_en_chroma(chroma_folklore2,   tema, k=k_folklore)
    fragmentos_folklore    = fragmentos_folklore1 + fragmentos_folklore2
    fragmentos_extendido = []
    for chroma_ext in chromas_extendido:
        fragmentos_extendido.extend(buscar_en_chroma(chroma_ext, tema, k=k_extendido))

    if loguear_etapa:
        loguear_etapa("Buscar en Chroma",
                      f"k_obra={k_obra} k_inf={k_influencies} "
                      f"k_folk={k_folklore} k_ext={k_extendido}",
                      f"obra={len(fragmentos_obra)}, "
                      f"inf={len(fragmentos_influencias)}, "
                      f"folk={len(fragmentos_folklore)}, "
                      f"ext={len(fragmentos_extendido)}")

    # 6. COMPACTAR FRAGMENTS
    texto_obra        = _compactar_fragmentos(fragmentos_obra,        MAX_CHARS_CONTEXTO_OBRA)
    texto_influencias = _compactar_fragmentos(fragmentos_influencias, MAX_CHARS_CONTEXTO_INFLUENCIAS)
    texto_folklore    = _compactar_fragmentos(fragmentos_folklore,    MAX_CHARS_CONTEXTO_FOLKLORE)
    texto_extendido   = _compactar_fragmentos(fragmentos_extendido,   MAX_CHARS_CONTEXTO_EXTENDIDO)

    # 7. FORMATAR PROMPT MAESTRO (amb pesos injectats)
    instrucciones_formateadas = prompt_maestro.format(
        estilo=perfil_estilistico,
        estructura=EstructuraFlexible(estructura),
        mezcla=texto_obra,
        influencias=texto_influencias,
        folklore=texto_folklore,
        tema=params.get("tema", ""),
        tono_extra=params.get("tono_extra", ""),
        restricciones=params.get("restricciones", ""),
        extension=params.get("extension", ""),
        # NOU: orientació quantitativa per al model
        peso_factual=params_ctx["peso_factual"],
        peso_literari=params_ctx["peso_literari"],
        peso_estil=params_ctx["peso_estil"],
    )
    if loguear_etapa:
        loguear_etapa("Formatear instrucciones", "Prompt maestro",
                      instrucciones_formateadas)

    # 8. CONSTRUIR CONTEXT EXTÈS
    CONTEXTO_EXTENDIDO = f"""PERFIL_ESTILISTICO:
{perfil_estilistico}

ESTRUCTURA_POETICA:
Número de estrofas: {estructura.get('num_estrofas', '?')}
Versos por estrofa: {estructura.get('versos_por_estrofa', '?')}
Tipo de verso: {estructura.get('tipo_verso', '?')}
Ritmo: {estructura.get('ritmo', '?')}
Progresión: {estructura.get('progresion', '?')}
Notas: {estructura.get('notas', '')}

PESOS_ADAPTATIUS:
α (ancoratge estilístic): {params_ctx['peso_estil']}
β (context literari):     {params_ctx['peso_literari']}
γ (factualitat externa):  {params_ctx['peso_factual']}

CONTEXTO_OBRA:
{texto_obra}

CONTEXTO_INFLUENCIAS:
{texto_influencias}

CONTEXTO_FOLKLORE:
{texto_folklore}

CONTEXTO_FACTUAL (actiu si γ > 0.15):
{contexto_factual or '— no requerit per a aquest perfil —'}

INSTRUCCIONES:
{instrucciones_formateadas}"""

    if loguear_etapa:
        loguear_etapa("Construir contexto extendido", "Contexto", CONTEXTO_EXTENDIDO)

    # 9. GENERACIÓ INICIAL
    prompt_generacion = f"{CONTEXTO_EXTENDIDO}\n\nTAREA:\nEscribe un poema sobre: {tema}"
    POEMA_INICIAL, modelo_usado_generacion = llamar_google_con_fallback(
        prompt_generacion, modelos=MODELOS_PRO)
    modelos_usados_info["generacion_pulido"] = modelo_usado_generacion
    if loguear_etapa:
        loguear_etapa("Generar poema inicial", prompt_generacion, POEMA_INICIAL)

    # 10. AVALUACIÓ INICIAL
    prompt_evaluacion = (
        f"{prompt_eval}\n\nPOEMA:\n{POEMA_INICIAL}"
        f"\n\nESTILO:\n{perfil_estilistico}\n\nTEMA:\n{tema}"
    )
    critica_str = llamar_deepseek(
        prompt_evaluacion,
        system_prompt=(
            "Eres un crítico literario experto. Responde estrictamente en JSON. "
            "Sé breve y conciso; limita problemas y sugerencias a lo esencial."
        ),
    )
    modelos_usados_info["evaluacion"] = _label_ultima_llamada("DeepSeek")
    CRITICA = _parsear_critica_json(critica_str)
    if loguear_etapa:
        loguear_etapa("Evaluar poema inicial", prompt_evaluacion, str(CRITICA))

    # 11. REESCRIPTURA ITERATIVA (màx 3, guardant el millor candidat)
    POEMA_CORREGIDO  = POEMA_INICIAL
    MILLOR_POEMA     = POEMA_INICIAL
    iteraciones      = 0
    max_iter         = 3

    while not (isinstance(CRITICA, dict) and CRITICA.get("ok", False)) and iteraciones < max_iter:
        probs = ", ".join(CRITICA.get("problemas", [])) if isinstance(CRITICA, dict) else ""
        sugs  = ", ".join(CRITICA.get("sugerencias", [])) if isinstance(CRITICA, dict) else ""
        prompt_reescritura = (
            f"{prompt_rewrite}\n\nPOEMA ORIGINAL:\n{POEMA_CORREGIDO}"
            f"\n\nPROBLEMAS:\n{probs}\n\nSUGERENCIAS:\n{sugs}"
            f"\n\nESTILO:\n{perfil_estilistico}"
        )
        POEMA_CANDIDAT, modelo_usado_reescritura = llamar_google_con_fallback(
            prompt_reescritura,
            system_prompt="Eres un editor de poesía experto.",
            modelos=MODELOS_FLASH,
        )
        modelos_usados_info["reescritura"] = modelo_usado_reescritura
        if loguear_etapa:
            loguear_etapa("Reescribir poema", prompt_reescritura, POEMA_CANDIDAT)

        prompt_eval_corr = (
            f"{prompt_eval}\n\nPOEMA:\n{POEMA_CANDIDAT}"
            f"\n\nESTILO:\n{perfil_estilistico}\n\nTEMA:\n{tema}"
        )
        critica_corr_str = llamar_deepseek(
            prompt_eval_corr,
            system_prompt=(
                "Eres un crítico literario experto. Responde estrictamente en JSON. "
                "Sé breve y conciso; limita problemas y sugerencias a lo esencial."
            ),
        )
        modelos_usados_info["evaluacion"] = _label_ultima_llamada("DeepSeek")
        CRITICA_CANDIDAT = _parsear_critica_json(critica_corr_str)
        if loguear_etapa:
            loguear_etapa("Evaluar poema corregido", prompt_eval_corr, str(CRITICA_CANDIDAT))

        # Guardar el millor candidat (evita backtracking a pitjor versió)
        critica_anterior_ok = isinstance(CRITICA, dict) and CRITICA.get("ok", False)
        critica_candidat_ok = isinstance(CRITICA_CANDIDAT, dict) and CRITICA_CANDIDAT.get("ok", False)
        if critica_candidat_ok or not critica_anterior_ok:
            POEMA_CORREGIDO = POEMA_CANDIDAT
            MILLOR_POEMA    = POEMA_CANDIDAT
            CRITICA         = CRITICA_CANDIDAT
        # Si el candidat empitjora, mantenim POEMA_CORREGIDO però actualitzem CRITICA
        else:
            CRITICA = CRITICA_CANDIDAT

        iteraciones += 1

    POEMA_CORREGIDO = MILLOR_POEMA

    # 12. PULIMENT FINAL
    prompt_pulido_final = (
        f"{CONTEXTO_EXTENDIDO}\n\nPOEMA PREVIO:\n{POEMA_CORREGIDO}"
        f"\n\nINSTRUCCIONES DE PULIDO:\n{prompt_pulido}"
    )
    POEMA_FINAL, modelo_usado_pulido = llamar_google_con_fallback(
        prompt_pulido_final, modelos=MODELOS_PRO)
    modelos_usados_info["generacion_pulido"] = modelo_usado_pulido
    if loguear_etapa:
        loguear_etapa("Pulido final", prompt_pulido_final, POEMA_FINAL)

    # 13. IMATGE (opcional)
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

    # 14. ÀUDIO (opcional)
    audio_url = audio_bytes = audio_estilo = None
    audio_task_id = audio_status = audio_error = None
    if params.get("crear_audio"):
        try:
            from utils_llamadas import generar_audio_poema_con_suno
            audio_resultado = generar_audio_poema_con_suno(
                poema_texto=POEMA_FINAL,
                contexto_poetico=CONTEXTO_EXTENDIDO,
                tema=params.get("tema", ""),
                tono_extra=params.get("tono_extra", ""),
                model_google=MODELOS_PRO[0],
                titulo=params.get("audio_title", ""),
                instrumental=params.get("audio_instrumental", False),
                wait_audio=params.get("audio_wait", True),
                duracion_segundos=params.get("audio_duracion_seg", 90),
            )
            audio_url      = audio_resultado.get("audio_url")
            audio_bytes    = audio_resultado.get("audio_bytes")
            audio_estilo   = audio_resultado.get("estilo_musical")
            audio_task_id  = audio_resultado.get("task_id")
            audio_status   = audio_resultado.get("status")
            if loguear_etapa:
                loguear_etapa("Generar audio", "Poema final + contexto",
                              str(audio_resultado))
        except Exception as e:
            audio_error = str(e)
            if loguear_etapa:
                loguear_etapa("Error generando audio", "Poema final + contexto",
                              audio_error)

    return {
        "poema_final":            POEMA_FINAL,
        "poema_inicial":          POEMA_INICIAL,
        "poema_corregido":        POEMA_CORREGIDO,
        "critica_final":          CRITICA,
        "contexto_extendido":     CONTEXTO_EXTENDIDO,
        "insumos_pulido": {
            "contexto_extendido": CONTEXTO_EXTENDIDO,
            "tema":               params.get("tema", ""),
            "tono_extra":         params.get("tono_extra", ""),
        },
        "historial_afinados":     [],
        "estructura":             estructura,
        "pesos":                  pesos,
        "params_ctx":             params_ctx,
        "perfil":                 perfil,
        "imagen":                 imagen,
        "audio_url":              audio_url,
        "audio_bytes":            audio_bytes,
        "audio_estilo":           audio_estilo,
        "audio_task_id":          audio_task_id,
        "audio_status":           audio_status,
        "audio_error":            audio_error,
        "iteraciones_reescritura": iteraciones,
        "modelos_usados": {
            "clasificacion_intencion": modelos_usados_info.get("clasificacion_intencion", "DeepSeek"),
            "generacion_estructura":   modelos_usados_info.get("generacion_estructura",   "DeepSeek"),
            "generacion_pulido":       modelos_usados_info.get("generacion_pulido",       "Gemini Pro"),
            "evaluacion":              modelos_usados_info.get("evaluacion",              "DeepSeek"),
            "reescritura":             modelos_usados_info.get("reescritura",             "Gemini Flash"),
            "busqueda_contexto":       "Brave Search" if params_ctx["γ"] > 0.15 else "No requerida",
        },
        "modelos_probados": get_model_attempt_history(),
        "groq_payload_ajustes": get_groq_call_history(),
    }


# ---------------------------------------------------------------------------
# PIPELINE DE REWORK (afinat sobre poema existent)
# ---------------------------------------------------------------------------

def ejecutar_rework_poetico(params):
    base = "."
    loguear_etapa = params.get("loguear_etapa")
    reset_groq_call_history()

    poema_base        = (params.get("poema_base") or "").strip()
    comentario_rework = (params.get("comentario_rework") or "").strip()
    if not poema_base:
        raise Exception("No hay poema base para afinar.")

    perfil_estilistico = leer_texto(f"{base}/estilo/perfil_estilistico_final.md")
    prompt_eval    = cargar_prompt(f"{base}/prompts/prompt_evaluacion.txt")
    prompt_rewrite = cargar_prompt(f"{base}/prompts/prompt_reescritura.txt")
    prompt_pulido  = cargar_prompt(f"{base}/prompts/prompt_pulido_final.txt")

    tema       = params.get("tema", "")
    tono_extra = params.get("tono_extra", "")
    contexto_extendido = params.get("contexto_extendido", "")

    # Fallback si el guardament és antic i no té contexto_extendido
    if not (contexto_extendido or "").strip():
        contexto_extendido = (
            f"PERFIL_ESTILISTICO:\n{perfil_estilistico}\n\n"
            f"CONTEXTO_MINIMO_REWORK:\n"
            f"TEMA: {tema}\n"
            f"TONO_EXTRA: {tono_extra}\n"
            "ORIGEN: rework sobre poema ja generat"
        )

    if loguear_etapa:
        loguear_etapa("Afinado - Entrada", f"Tema: {tema}",
                      f"Comentario: {comentario_rework}\n\nPoema base:\n{poema_base}")

    modelos_usados_info = {}

    # 1. REWORK inicial guiat pel comentari de l'usuari
    prompt_rework = (
        f"{prompt_rewrite}\n\n"
        f"COMENTARIO DEL USUARIO (prioritario):\n{comentario_rework or 'Sin comentario explícito.'}\n\n"
        f"POEMA ORIGINAL:\n{poema_base}\n\n"
        f"PROBLEMAS:\nDerivados del comentario del usuario\n\n"
        f"SUGERENCIAS:\nAplica exactamente la intención del comentario manteniendo coherencia y calidad poética\n\n"
        f"ESTILO:\n{perfil_estilistico}"
    )
    POEMA_CORREGIDO, modelo_usado_rework = llamar_google_con_fallback(
        prompt_rework,
        system_prompt="Eres un editor de poesía experto.",
        modelos=MODELOS_PRO,
    )
    modelos_usados_info["reescritura"] = modelo_usado_rework
    if loguear_etapa:
        loguear_etapa("Afinado - Rework inicial", prompt_rework, POEMA_CORREGIDO)

    # 2. AVALUACIÓ
    prompt_evaluacion = (
        f"{prompt_eval}\n\nPOEMA:\n{POEMA_CORREGIDO}"
        f"\n\nESTILO:\n{perfil_estilistico}\n\nTEMA:\n{tema}"
    )
    critica_str = llamar_deepseek(
        prompt_evaluacion,
        system_prompt=(
            "Eres un crítico literario experto. Responde estrictamente en JSON. "
            "Sé breve y conciso; limita problemas y sugerencias a lo esencial."
        ),
    )
    modelos_usados_info["evaluacion"] = _label_ultima_llamada("DeepSeek")
    CRITICA = _parsear_critica_json(critica_str)
    if loguear_etapa:
        loguear_etapa("Afinado - Evaluación", prompt_evaluacion, str(CRITICA))

    # 3. REESCRIPTURA ITERATIVA (amb backtracking protegit)
    MILLOR_POEMA = POEMA_CORREGIDO
    iteraciones  = 0
    max_iter     = 3

    while not CRITICA.get("ok", False) and iteraciones < max_iter:
        probs = ", ".join(CRITICA.get("problemas", []))
        sugs  = ", ".join(CRITICA.get("sugerencias", []))
        prompt_reescritura = (
            f"{prompt_rewrite}\n\n"
            f"COMENTARIO DEL USUARIO (mantener intención):\n{comentario_rework or 'Sin comentario explícito.'}\n\n"
            f"POEMA ORIGINAL:\n{POEMA_CORREGIDO}\n\n"
            f"PROBLEMAS:\n{probs}\n\nSUGERENCIAS:\n{sugs}\n\n"
            f"ESTILO:\n{perfil_estilistico}"
        )
        POEMA_CANDIDAT, modelo_reescritura = llamar_google_con_fallback(
            prompt_reescritura,
            system_prompt="Eres un editor de poesía experto.",
            modelos=MODELOS_PRO,
        )
        modelos_usados_info["reescritura"] = modelo_reescritura
        if loguear_etapa:
            loguear_etapa("Afinado - Reescritura", prompt_reescritura, POEMA_CANDIDAT)

        prompt_eval_it = (
            f"{prompt_eval}\n\nPOEMA:\n{POEMA_CANDIDAT}"
            f"\n\nESTILO:\n{perfil_estilistico}\n\nTEMA:\n{tema}"
        )
        critica_it_str = llamar_deepseek(
            prompt_eval_it,
            system_prompt=(
                "Eres un crítico literario experto. Responde estrictamente en JSON. "
                "Sé breve y conciso; limita problemas y sugerencias a lo esencial."
            ),
        )
        modelos_usados_info["evaluacion"] = _label_ultima_llamada("DeepSeek")
        CRITICA_IT = _parsear_critica_json(critica_it_str)
        if loguear_etapa:
            loguear_etapa("Afinado - Reevaluación", prompt_eval_it, str(CRITICA_IT))

        critica_anterior_ok  = CRITICA.get("ok", False)
        critica_candidat_ok  = CRITICA_IT.get("ok", False)
        if critica_candidat_ok or not critica_anterior_ok:
            POEMA_CORREGIDO = POEMA_CANDIDAT
            MILLOR_POEMA    = POEMA_CANDIDAT
            CRITICA         = CRITICA_IT
        else:
            CRITICA = CRITICA_IT

        iteraciones += 1

    POEMA_CORREGIDO = MILLOR_POEMA

    # 4. PULIMENT FINAL
    prompt_pulido_final = (
        f"{contexto_extendido}\n\nPOEMA PREVIO:\n{POEMA_CORREGIDO}"
        f"\n\nINSTRUCCIONES DE PULIDO:\n{prompt_pulido}"
    )
    POEMA_FINAL, modelo_pulido = llamar_google_con_fallback(
        prompt_pulido_final, modelos=MODELOS_PRO)
    modelos_usados_info["generacion_pulido"] = modelo_pulido
    if loguear_etapa:
        loguear_etapa("Afinado - Pulido final", prompt_pulido_final, POEMA_FINAL)

    return {
        "poema_final":             POEMA_FINAL,
        "poema_inicial":           poema_base,
        "poema_corregido":         POEMA_CORREGIDO,
        "critica_final":           CRITICA,
        "contexto_extendido":      contexto_extendido,
        "comentario_rework":       comentario_rework,
        "insumos_pulido": {
            "contexto_extendido":  contexto_extendido,
            "tema":                tema,
            "tono_extra":          tono_extra,
        },
        "iteraciones_reescritura": iteraciones,
        "modelos_usados": {
            "generacion_pulido":   modelos_usados_info.get("generacion_pulido", "Gemini Pro"),
            "evaluacion":          modelos_usados_info.get("evaluacion",         "DeepSeek"),
            "reescritura":         modelos_usados_info.get("reescritura",        "Gemini Pro"),
            "busqueda_contexto":   "No requerida",
        },
        "modelos_probados": get_model_attempt_history(),
        "groq_payload_ajustes": get_groq_call_history(),
    }