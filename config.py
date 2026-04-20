# config.py

import os
from dotenv import load_dotenv
try:
    import streamlit as st
except ImportError:
    st = None

load_dotenv()

def get_config(key, default=None):
    # 1. Prioridad: Streamlit Secrets (Nube / App Mode)
    if st is not None:
        try:
            if hasattr(st, "secrets") and key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass

    # 2. Fallback: Variable de entorno (Local .env)
    val = os.getenv(key)
    if val is not None:
        return val
            
    return default

GROQ_API_KEY = get_config("GROQ_API_KEY")
GROQ_MODEL = get_config("GROQ_MODEL", "llama-3.3-70b-versatile")
REWORK_RETRIES = int(get_config("REWORK_RETRIES", "3"))
DIALOG_RETRIES = int(get_config("DIALOG_RETRIES", "2"))
GOOGLE_API_KEY = get_config("GOOGLE_API_KEY")
GOOGLE_MODEL = get_config("GOOGLE_MODEL", "gemini-2.0-pro")
GOOGLE_FAST_MODEL = get_config("GOOGLE_FAST_MODEL", "gemini-2.0-flash")
# Nota: Usaremos los modelos Lyria que tienes disponibles.
# Para generación de música, se requiere acceso a APIs específicas.
GOOGLE_MUSIC_MODEL = get_config("GOOGLE_MUSIC_MODEL", "models/lyria-3-pro-preview")
GOOGLE_MUSIC_VOICE = get_config("GOOGLE_MUSIC_VOICE", "Kore")
BRAVE_SEARCH_API_KEY = get_config("BRAVE_SEARCH_API_KEY")
DEEPSEEK_API_KEY = get_config("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = get_config("DEEPSEEK_MODEL", "deepseek-chat")