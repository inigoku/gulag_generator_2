import sys
import os

# Aseguramos que se pueda importar desde el directorio actual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generar_poema import ejecutar_pipeline_poetico

def main():
    # Valores por defecto para los parámetros (similares a construir_prompt_maestro)
    default_params = {
        "estilo": "Estilo libre pero lírico",
        "tema": "La emoción del básquet",
        "tono_extra": "Épico y apasionado",
        "restricciones": "Sin rima consonante forzada, sin referencias tecnológicas",
        "extension": 20
    }
    
    print(f"=== Ejecutando Pipeline Poético ===")
    print(f"Tema: {default_params['tema']}")
    
    resultado = ejecutar_pipeline_poetico(default_params)
    
    print("\n=== POEMA FINAL ===")
    print(resultado["poema_final"])

if __name__ == "__main__":
    main()