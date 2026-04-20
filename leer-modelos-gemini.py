import google.generativeai as genai
import os
from dotenv import load_dotenv

# Asegúrate de tener tu clave de API configurada como una variable de entorno
# Por ejemplo: export GOOGLE_API_KEY="TU_API_KEY"
load_dotenv()  # Carga las variables de entorno desde un archivo .env si existe
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    print("Error: La variable de entorno GOOGLE_API_KEY no está configurada.")
    exit()


print("--- Modelos disponibles que soportan 'generateContent' ---")
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(f"Nombre del modelo: {m.name}")
    print(f"  - Descripción: {m.description}\n")
