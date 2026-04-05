import streamlit as st
import sys
import os
import re
import json
import datetime
from pathlib import Path

import requests

# Configuración de página debe ser el primer comando de Streamlit para evitar errores
st.set_page_config(page_title="Generador de Poesía V2", page_icon="✒️", layout="centered")

# Aseguramos que se pueda importar desde el directorio actual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generar_poema import ejecutar_pipeline_poetico
from utils_llamadas import consultar_estado_tarea_suno, generate_from_poem, generar_audio_poema_con_suno

BASE_DIR = Path(__file__).resolve().parent
GUARDADOS_DIR = BASE_DIR / "guardados"


def _slug(texto):
    limpio = re.sub(r"[^a-zA-Z0-9_-]+", "-", (texto or "").strip()).strip("-")
    return limpio[:50] or "poema"


def _listar_guardados():
    if not GUARDADOS_DIR.exists():
        return []
    return sorted([p.name for p in GUARDADOS_DIR.iterdir() if p.is_dir()], reverse=True)


def _descargar_bytes(url):
    if not isinstance(url, str) or not url.startswith("http"):
        return None
    resp = requests.get(url, timeout=60)
    if resp.status_code >= 400:
        return None
    return resp.content if resp.content else None


def _guardar_bundle(resultado, params, nombre):
    GUARDADOS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta = GUARDADOS_DIR / f"{stamp}_{_slug(nombre)}"
    carpeta.mkdir(parents=True, exist_ok=False)

    poema = (resultado or {}).get("poema_final", "")
    (carpeta / "poema.txt").write_text(poema or "", encoding="utf-8")

    imagen_path = None
    imagen_data = (resultado or {}).get("imagen")
    if isinstance(imagen_data, (bytes, bytearray)):
        imagen_path = "imagen.bin"
        (carpeta / imagen_path).write_bytes(bytes(imagen_data))

    audio_path = None
    audio_url = (resultado or {}).get("audio_url")
    audio_bytes = (resultado or {}).get("audio_bytes")
    if isinstance(audio_bytes, (bytes, bytearray)):
        audio_path = "audio.mp3"
        (carpeta / audio_path).write_bytes(bytes(audio_bytes))
    elif audio_url:
        descargado = _descargar_bytes(audio_url)
        if descargado:
            audio_path = "audio.mp3"
            (carpeta / audio_path).write_bytes(descargado)

    metadata = {
        "nombre": nombre,
        "creado_en": stamp,
        "tema": (params or {}).get("tema", ""),
        "tono_extra": (params or {}).get("tono_extra", ""),
        "poema_path": "poema.txt",
        "imagen_path": imagen_path,
        "audio_path": audio_path,
        "audio_url": audio_url,
        "audio_status": (resultado or {}).get("audio_status", ""),
        "audio_task_id": (resultado or {}).get("audio_task_id", "")
    }
    (carpeta / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return carpeta.name


def _recuperar_bundle(bundle_id):
    carpeta = GUARDADOS_DIR / bundle_id
    if not carpeta.exists():
        raise Exception("Guardado no encontrado")

    metadata_path = carpeta / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    poema_path = carpeta / (metadata.get("poema_path") or "poema.txt")
    poema = poema_path.read_text(encoding="utf-8") if poema_path.exists() else ""

    imagen = None
    img_rel = metadata.get("imagen_path")
    if img_rel:
        img_path = carpeta / img_rel
        if img_path.exists():
            imagen = img_path.read_bytes()

    audio_bytes = None
    aud_rel = metadata.get("audio_path")
    if aud_rel:
        aud_path = carpeta / aud_rel
        if aud_path.exists():
            audio_bytes = aud_path.read_bytes()

    return {
        "poema_final": poema,
        "imagen": imagen,
        "audio_bytes": audio_bytes,
        "audio_url": metadata.get("audio_url"),
        "audio_status": metadata.get("audio_status"),
        "audio_task_id": metadata.get("audio_task_id")
    }, metadata

def main():
    if "consulta_audio_status" not in st.session_state:
        st.session_state.consulta_audio_status = ""
    if "consulta_audio_url" not in st.session_state:
        st.session_state.consulta_audio_url = ""
    if "consulta_task_id" not in st.session_state:
        st.session_state.consulta_task_id = ""
    if "consulta_respuesta" not in st.session_state:
        st.session_state.consulta_respuesta = {}
    if "ultimo_resultado" not in st.session_state:
        st.session_state.ultimo_resultado = {}
    if "ultimo_params" not in st.session_state:
        st.session_state.ultimo_params = {}
    if "ultimo_bundle_id" not in st.session_state:
        st.session_state.ultimo_bundle_id = ""

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
            crear_audio = st.checkbox("Generar audio (Suno)", value=False)

        audio_title = ""
        audio_instrumental = False
        audio_wait = True
        audio_duracion_seg = 90
        if crear_audio:
            with st.expander("Opciones avanzadas de audio", expanded=False):
                audio_title = st.text_input("Título de la canción (opcional)", value="")
                audio_instrumental = st.checkbox("Modo instrumental", value=False)
                audio_wait = st.checkbox("Esperar audio final en la respuesta", value=True)
                audio_duracion_seg = st.slider("Duración objetivo (segundos)", min_value=30, max_value=240, value=90, step=10)
        
        restricciones = st.text_area("Restricciones", value="Sin rima consonante forzada, sin referencias tecnológicas")

    with st.expander("Consultar estado de audio (taskId)", expanded=False):
        consulta_task_id = st.text_input("Task ID de Suno", value="")
        if st.button("Consultar estado", use_container_width=True):
            if not consulta_task_id.strip():
                st.warning("Introduce un taskId válido.")
            else:
                with st.spinner("Consultando estado en Suno..."):
                    try:
                        estado = consultar_estado_tarea_suno(consulta_task_id.strip())
                        st.session_state.consulta_audio_status = str(estado.get("status", "unknown"))
                        st.session_state.consulta_audio_url = str(estado.get("audio_url") or "")
                        st.session_state.consulta_task_id = str(estado.get("task_id", consulta_task_id.strip()))
                        st.session_state.consulta_respuesta = estado.get("respuesta_suno", {})
                    except Exception as e:
                        st.error(f"Error consultando estado: {e}")

        if st.session_state.consulta_task_id:
            st.write(f"Estado: {st.session_state.consulta_audio_status or 'unknown'}")
            st.caption(f"Task ID: {st.session_state.consulta_task_id}")

        puede_escuchar = (
            st.session_state.consulta_audio_status.upper() == "SUCCESS"
            and bool(st.session_state.consulta_audio_url)
        )
        if st.button("Escuchar canción", use_container_width=True, disabled=not puede_escuchar):
            st.audio(st.session_state.consulta_audio_url, format="audio/mpeg")
            st.markdown(f"[Abrir audio en nueva pestaña]({st.session_state.consulta_audio_url})")

        if st.session_state.consulta_respuesta:
            st.json(st.session_state.consulta_respuesta)

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
                "crear_imagen": crear_imagen,
                "crear_audio": crear_audio,
                "audio_title": audio_title,
                "audio_instrumental": audio_instrumental,
                "audio_wait": audio_wait,
                "audio_duracion_seg": audio_duracion_seg
            }

            # --- Proceso de Generación ---
            with st.spinner("🤖 El agente está consultando la obra, escribiendo y puliendo..."):
                
                try:
                    resultado = ejecutar_pipeline_poetico(params)
                    st.session_state.ultimo_resultado = resultado
                    st.session_state.ultimo_params = params
                    st.success("¡Poema generado con éxito!")
                    
                except Exception as e:
                    st.error(f"❌ Ocurrió un error: {e}")

    resultado_actual = st.session_state.ultimo_resultado
    if resultado_actual and resultado_actual.get("poema_final"):
        st.subheader("Poema Final")
        st.text_area("Resultado", value=resultado_actual["poema_final"], height=500)

        if resultado_actual.get("imagen"):
            st.subheader("Imagen Generada")
            st.image(resultado_actual["imagen"], caption="Imagen generada a partir del poema.")

        if resultado_actual.get("audio_url"):
            st.subheader("Audio Generado")
            st.caption(f"Estilo sugerido por Gemini: {resultado_actual.get('audio_estilo', 'No disponible')}")
            st.audio(resultado_actual["audio_url"], format="audio/mpeg")
            st.markdown(f"[Abrir audio en nueva pestaña]({resultado_actual['audio_url']})")
        elif resultado_actual.get("audio_bytes"):
            st.subheader("Audio Recuperado")
            st.audio(resultado_actual["audio_bytes"], format="audio/mpeg")
        elif resultado_actual.get("audio_task_id"):
            st.info(
                f"Audio en proceso. Task ID: {resultado_actual.get('audio_task_id')} | "
                f"Estado: {resultado_actual.get('audio_status', 'processing')}"
            )
        elif resultado_actual.get("audio_error"):
            st.warning(f"No se pudo generar audio: {resultado_actual['audio_error']}")

        st.markdown("### Acciones sobre el poema actual")
        col_img, col_audio = st.columns(2)

        with col_img:
            label_img = "Regenerar imagen" if resultado_actual.get("imagen") else "Generar imagen"
            if st.button(label_img, use_container_width=True):
                with st.spinner("Generando imagen desde el poema actual..."):
                    try:
                        nueva_imagen = generate_from_poem(resultado_actual.get("poema_final", ""))
                        st.session_state.ultimo_resultado["imagen"] = nueva_imagen
                        st.success("Imagen generada correctamente.")
                    except Exception as e:
                        st.error(f"Error generando imagen: {e}")

        with col_audio:
            label_audio = "Regenerar audio" if (resultado_actual.get("audio_url") or resultado_actual.get("audio_task_id")) else "Generar audio"
            if st.button(label_audio, use_container_width=True):
                with st.spinner("Generando audio desde el poema actual..."):
                    try:
                        params_ultimo = st.session_state.ultimo_params or {}
                        contexto_audio = resultado_actual.get("contexto_extendido") or (
                            f"TEMA: {params_ultimo.get('tema', '')}\nTONO: {params_ultimo.get('tono_extra', '')}"
                        )
                        audio_resultado = generar_audio_poema_con_suno(
                            poema_texto=resultado_actual.get("poema_final", ""),
                            contexto_poetico=contexto_audio,
                            tema=params_ultimo.get("tema", ""),
                            tono_extra=params_ultimo.get("tono_extra", ""),
                            model_google=params_ultimo.get("google_model"),
                            titulo=audio_title or params_ultimo.get("audio_title", ""),
                            instrumental=audio_instrumental,
                            wait_audio=audio_wait,
                            duracion_segundos=audio_duracion_seg
                        )
                        st.session_state.ultimo_resultado["audio_url"] = audio_resultado.get("audio_url")
                        st.session_state.ultimo_resultado["audio_estilo"] = audio_resultado.get("estilo_musical")
                        st.session_state.ultimo_resultado["audio_task_id"] = audio_resultado.get("task_id")
                        st.session_state.ultimo_resultado["audio_status"] = audio_resultado.get("status")
                        st.session_state.ultimo_resultado["audio_error"] = None
                        st.success("Audio solicitado correctamente.")
                    except Exception as e:
                        st.session_state.ultimo_resultado["audio_error"] = str(e)
                        st.error(f"Error generando audio: {e}")

        st.markdown("### Guardar y recuperar")
        col_save, col_load = st.columns(2)

        with col_save:
            nombre_guardado = st.text_input("Nombre del guardado", value="poema")
            if st.button("Grabar poema + imagen + canción", use_container_width=True):
                try:
                    bundle_id = _guardar_bundle(
                        resultado=resultado_actual,
                        params=st.session_state.ultimo_params,
                        nombre=nombre_guardado
                    )
                    st.session_state.ultimo_bundle_id = bundle_id
                    st.success(f"Guardado creado: {bundle_id}")
                except Exception as e:
                    st.error(f"Error al grabar: {e}")

        with col_load:
            opciones = _listar_guardados()
            seleccionado = None
            if opciones:
                seleccionado = st.selectbox("Guardados disponibles", options=opciones, index=0)
            else:
                st.caption("No hay guardados todavía.")

            if st.button("Recuperar poema + imagen + canción", use_container_width=True, disabled=not bool(opciones)):
                try:
                    recuperado, meta = _recuperar_bundle(seleccionado)
                    actual = dict(st.session_state.ultimo_resultado)
                    actual.update(recuperado)
                    st.session_state.ultimo_resultado = actual
                    st.session_state.ultimo_bundle_id = seleccionado
                    st.success(f"Guardado recuperado: {seleccionado}")
                    if meta.get("audio_url"):
                        st.caption("Incluye URL de audio original y copia local si estaba disponible.")
                except Exception as e:
                    st.error(f"Error al recuperar: {e}")

        if st.session_state.ultimo_bundle_id:
            st.caption(f"Último guardado usado: {st.session_state.ultimo_bundle_id}")

        with st.expander("Ver detalles del proceso"):
            st.markdown("**1. Poema Inicial (Gemini + RAG):**")
            st.text(resultado_actual.get("poema_inicial", ""))
            
            st.markdown("**2. Crítica (Groq):**")
            st.json(resultado_actual.get("critica_final", {}))
            
            st.markdown("**3. Poema Corregido:**")
            st.text(resultado_actual.get("poema_corregido", ""))

            if resultado_actual.get("audio_estilo"):
                st.markdown("**4. Estilo musical sugerido (Gemini):**")
                st.text(resultado_actual.get("audio_estilo", ""))

if __name__ == "__main__":
    main()