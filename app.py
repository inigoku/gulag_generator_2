import streamlit as st
import sys
import os

# Configuración de página debe ser el primer comando de Streamlit para evitar errores
st.set_page_config(page_title="Generador de Poesía V2", page_icon="✒️", layout="centered")

# Aseguramos que se pueda importar desde el directorio actual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generar_poema import ejecutar_pipeline_poetico

def main():
    st.title("Generador de Poesía V2: Sindar")
    st.markdown("Configura los parámetros y genera poemas utilizando el pipeline poético (RAG + Crítica + Pulido).")

    # --- Barra Lateral: Configuración de Modelos ---
    with st.sidebar:
        st.header("⚙️ Configuración de Modelos")
        groq_model = st.selectbox(
            "Modelo Groq (Crítica/Reescritura)",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
            index=0
        )
        google_model = st.selectbox(
            "Modelo Google (Generación/Pulido)",
            ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-pro", "gemini-2.0-flash"],
            index=0
        )

    # --- Configuración de Parámetros ---
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            # Valores por defecto tomados de main.py
            estilo = st.text_input("Estilo", value="Estilo libre pero lírico")
            extension = st.selectbox("Extensión", ["muy corta", "corta", "media", "larga"], index=2)

        with col2:
            tema = st.text_input("Tema", value="La emoción del básquet")
            tono_extra = st.text_input("Tono Extra", value="Épico y apasionado")
            crear_imagen = st.checkbox("Crear imagen", value=False)
        
        restricciones = st.text_area("Restricciones", value="Sin rima consonante forzada, sin referencias tecnológicas")

    # --- Botón de Generación ---
    st.markdown("---")
    if st.button("Generar Poema", type="primary", use_container_width=True):
        if not tema:
            st.warning("⚠️ Por favor, escribe un tema para el poema.")
        else:
            params = {
                "estilo": estilo,
                "tema": tema,
                "tono_extra": tono_extra,
                "restricciones": restricciones,
                "extension": extension,
                "groq_model": groq_model,
                "google_model": google_model,
                "crear_imagen": crear_imagen
            }

            # --- Proceso de Generación ---
            with st.spinner("🤖 El agente está consultando la obra, escribiendo y puliendo..."):
                
                try:
                    resultado = ejecutar_pipeline_poetico(params)
                    
                    st.success("¡Poema generado con éxito!")
                    
                    st.subheader("Poema Final")
                    st.text_area("Resultado", value=resultado["poema_final"], height=500)

                    if resultado.get("imagen"):
                        st.subheader("Imagen Generada")
                        st.image(resultado["imagen"], caption="Imagen generada a partir del poema.")

                    with st.expander("Ver detalles del proceso"):
                        st.markdown("**1. Poema Inicial (Gemini + RAG):**")
                        st.text(resultado.get("poema_inicial", ""))
                        
                        st.markdown("**2. Crítica (Groq):**")
                        st.json(resultado.get("critica_final", {}))
                        
                        st.markdown("**3. Poema Corregido:**")
                        st.text(resultado.get("poema_corregido", ""))
                    
                except Exception as e:
                    st.error(f"❌ Ocurrió un error: {e}")

if __name__ == "__main__":
    main()