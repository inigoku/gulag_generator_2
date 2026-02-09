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
GROQ_MODEL = get_config("GROQ_MODEL", "qwen/qwen3-32b")
REWORK_RETRIES = int(get_config("REWORK_RETRIES", "3"))
DIALOG_RETRIES = int(get_config("DIALOG_RETRIES", "2"))
GOOGLE_API_KEY = get_config("GOOGLE_API_KEY")
GOOGLE_MODEL = get_config("GOOGLE_MODEL", "gemma-3-4b-it")
BRAVE_SEARCH_API_KEY = get_config("BRAVE_SEARCH_API_KEY")
DEEPSEEK_API_KEY = get_config("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = get_config("DEEPSEEK_MODEL", "deepseek-chat")