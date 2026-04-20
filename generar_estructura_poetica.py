import json
from utils_llamadas import llamar_deepseek

def generar_estructura_poetica(perfil):
    """
    Genera una estructura poética flexible basada en el perfil y las restricciones.
    Si el usuario pide una forma clásica (soneto, haiku, etc.), la respetará.
    Si no, generará una estructura orgánica.
    """

    prompt = f"""
Eres un arquitecto poético. Tu objetivo es definir la estructura formal de un poema basándote en las instrucciones del usuario.

PERFIL DETECTADO:
- Intención: {perfil.get("intencion_poetica", "")}
- Tono: {perfil.get("tono_emocional", "")}
- Extensión solicitada: {perfil.get("extension", "")}
- Estilo extra: {perfil.get("estilo_extra", "")}
- Restricciones explícitas: {perfil.get("restricciones", "")}

REGLAS:
1. Si en las restricciones o en el estilo se pide EXPLÍCITAMENTE una forma métrica o estrófica clásica (ej. "soneto", "haiku", "décima", "romance", "tercetos encadenados"), DEBES adaptar la estructura estrictamente a esa forma. Por ejemplo, para un soneto: num_estrofas="4", versos_por_estrofa="2 cuartetos y 2 tercetos", tipo_verso="endecasílabo", ritmo="clásico".
2. Si NO se pide una forma explícita, genera una estructura de verso libre o forma orgánica adecuada a la extensión solicitada y a la intención poética.

Devuelve ÚNICAMENTE un objeto JSON válido con las siguientes claves (todas deben ser strings descriptivos):
{{
    "num_estrofas": "descripción del número y tipo de secciones",
    "versos_por_estrofa": "cómo se distribuyen los versos",
    "tipo_verso": "clásico o libre, métrica si aplica",
    "ritmo": "descripción de la cadencia",
    "progresion": "cómo evoluciona el poema",
    "notas": "cualquier nota técnica sobre la estructura"
}}
"""

    try:
        respuesta = llamar_deepseek(
            prompt,
            system_prompt="Eres un experto en teoría métrica y diseño estructural de poesía. Responde estrictamente en JSON. Sé breve y conciso; usa descripciones cortas y no añadas texto fuera del JSON."
        )
        
        start = respuesta.find("{")
        end = respuesta.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(respuesta[start:end])
    except Exception as e:
        print(f"Error generando estructura con IA: {e}")

    # Fallback orgánico
    return {
        "num_estrofas": "2–3 secciones amplias",
        "versos_por_estrofa": "versos de longitud variable",
        "tipo_verso": "verso libre",
        "ritmo": "respiración irregular pero armónica",
        "progresion": "movimiento de lo íntimo a lo abierto",
        "notas": "estructura libre si no se especifica forma"
    }
