import json
from utils_llamadas import llamar_groq, cargar_prompt

def clasificar_intencion_poetica(tema, estilo_extra, tono_extra, restricciones, extension):
    """
    Llama al clasificador de intención poética usando el prompt especializado.
    Devuelve un diccionario con la clasificación completa.
    """

    prompt_base = cargar_prompt("./prompts/prompt_clasificador_tema.txt")

    prompt = prompt_base.replace("{tema}", tema)\
                        .replace("{estilo_extra}", estilo_extra)\
                        .replace("{tono_extra}", tono_extra)\
                        .replace("{restricciones}", restricciones)\
                        .replace("{extension}", extension)

    # Puedes elegir Groq o Google. Aquí uso Groq por consistencia.
    respuesta = llamar_groq(
        prompt,
        system_prompt="Eres un analista literario experto. Responde estrictamente en JSON."
    )

    # Extraer JSON de la respuesta
    try:
        start = respuesta.find("{")
        end = respuesta.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(respuesta[start:end])
    except:
        pass

    # Fallback seguro
    return {
        "categoria": "conceptual",
        "tono_emocional": "sereno",
        "nivel_abstraccion": "media",
        "grado_factualidad": "baja",
        "densidad_metaforica": "media",
        "intencion_poetica": "evocativa",
        "estilo_extra": estilo_extra,
        "restricciones": restricciones,
        "extension": extension,
        "estructura": {}
    }
