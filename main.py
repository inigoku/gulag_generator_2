import sys
import os
import datetime

# Aseguramos que se pueda importar desde el directorio actual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generar_poema import ejecutar_pipeline_poetico

def obtener_log_path():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"./logs/poetico_{timestamp}.log"

LOG_PATH = obtener_log_path()
flow_logs = []

def loguear_etapa(etapa, prompt, respuesta):
    prompt_chars = len(prompt or "")
    entry = (
        f"\n[{datetime.datetime.now().isoformat()}] ETAPA: {etapa}\n"
        f"PROMPT: [oculto en trazas, longitud={prompt_chars} chars]\n"
        f"RESPUESTA:\n{respuesta}\n{'-'*60}\n"
    )
    flow_logs.append(entry)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)

def main():
    default_params = {
        "estilo": "Estilo libre pero lírico",
        "tema": "Alzheimer, memoria y desesperanza",
        "tono_extra": "Triste pero esperanzador",
        "restricciones": "Sin rima consonante forzada, sin referencias tecnológicas, sin palabros como poiesis, agencimientos, glosolalia, etc. Evitar clichés. No usar palabras como 'olvido' o 'recuerdo'.",
        "extension": "media",
        "loguear_etapa": loguear_etapa
    }

    resultado = ejecutar_pipeline_poetico(default_params)

    print("\n=== POEMA FINAL ===")
    print(resultado["poema_final"])

if __name__ == "__main__":
    main()