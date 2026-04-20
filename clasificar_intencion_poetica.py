import json
from utils_llamadas import llamar_deepseek, cargar_prompt

import os

def clasificar_intencion_poetica(tema, estilo_extra, tono_extra, restricciones, extension):
    """
    Llama al clasificador de intención poética usando el prompt especializado.
    Devuelve un diccionario con la clasificación completa.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(base_dir, "prompts", "prompt_clasificacion_tema.txt")
    prompt_base = cargar_prompt(prompt_path)

    prompt = prompt_base.replace("{tema}", tema)\
                        .replace("{estilo_extra}", estilo_extra)\
                        .replace("{tono_extra}", tono_extra)\
                        .replace("{restricciones}", restricciones)\
                        .replace("{extension}", extension)

    respuesta = llamar_deepseek(
        prompt,
        system_prompt="Eres un analista literario experto. Responde estrictamente en JSON. Sé breve y conciso; no añadas texto fuera del JSON."
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
