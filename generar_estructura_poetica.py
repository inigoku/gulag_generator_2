import random

def generar_estructura_poetica(perfil):
    """
    Genera una estructura poética flexible basada en la intención.
    En lugar de números rígidos, devuelve descripciones orientativas.
    """

    intencion = perfil.get("intencion", "")
    tono = perfil.get("tono_extra", "")
    extension = perfil.get("extension", "")

    # Opciones de estructura flexible
    estructuras = [
        {
            "num_estrofas": "2–3 secciones amplias",
            "versos_por_estrofa": "versos de longitud variable",
            "tipo_verso": "verso libre",
            "ritmo": "respiración irregular pero armónica",
            "progresion": "movimiento de lo íntimo a lo abierto",
            "notas": "puede romperse la estructura si el poema lo pide"
        },
        {
            "num_estrofas": "1 bloque continuo",
            "versos_por_estrofa": "flujo sin divisiones estrictas",
            "tipo_verso": "verso libre con variaciones",
            "ritmo": "cadencia suave y orgánica",
            "progresion": "crecimiento gradual hacia una imagen final",
            "notas": "permitido cortar o expandir según emoción"
        },
        {
            "num_estrofas": "3 momentos diferenciados",
            "versos_por_estrofa": "versos cortos mezclados con algunos largos",
            "tipo_verso": "verso libre con respiraciones marcadas",
            "ritmo": "alternancia entre tensión y calma",
            "progresion": "viaje emocional en tres fases",
            "notas": "la estructura puede deformarse si el tono lo exige"
        }
    ]

    # Selección adaptativa
    estructura = random.choice(estructuras)

    return estructura
