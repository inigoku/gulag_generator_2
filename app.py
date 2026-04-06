import streamlit as st
import sys
import os
import re
import json
import datetime
from pathlib import Path

import requests

# Configuración de página debe ser el primer comando de Streamlit para evitar errores
st.set_page_config(page_title="Generador de Poesía V2", page_icon="✒️", layout="wide")

# Aseguramos que se pueda importar desde el directorio actual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generar_poema import ejecutar_pipeline_poetico, ejecutar_rework_poetico
from utils_llamadas import (
    generate_from_poem,
    generar_audio_poema_con_suno,
    generar_estilo_musical_desde_contexto,
)

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

    # Guardamos un snapshot serializable del resultado para poder restaurar
    # también el proceso (poema inicial, crítica, poema corregido, etc.).
    snapshot_resultado = {
        "poema_final": (resultado or {}).get("poema_final", ""),
        "poema_inicial": (resultado or {}).get("poema_inicial", ""),
        "poema_corregido": (resultado or {}).get("poema_corregido", ""),
        "critica_final": (resultado or {}).get("critica_final", {}),
        "contexto_extendido": (resultado or {}).get("contexto_extendido", ""),
        "insumos_pulido": (resultado or {}).get("insumos_pulido", {}),
        "historial_afinados": (resultado or {}).get("historial_afinados", []),
        "audio_url": (resultado or {}).get("audio_url"),
        "audio_status": (resultado or {}).get("audio_status", ""),
        "audio_task_id": (resultado or {}).get("audio_task_id", ""),
        "audio_estilo": (resultado or {}).get("audio_estilo", ""),
        "audio_error": (resultado or {}).get("audio_error"),
    }

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
        "audio_task_id": (resultado or {}).get("audio_task_id", ""),
        "params": params or {},
        "snapshot_resultado": snapshot_resultado,
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

    snapshot = metadata.get("snapshot_resultado") or {}
    metadata["es_legacy"] = "snapshot_resultado" not in metadata

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

    recuperado = {
        "poema_final": snapshot.get("poema_final", poema),
        "poema_inicial": snapshot.get("poema_inicial", ""),
        "poema_corregido": snapshot.get("poema_corregido", ""),
        "critica_final": snapshot.get("critica_final", {}),
        "contexto_extendido": snapshot.get("contexto_extendido", ""),
        "insumos_pulido": snapshot.get("insumos_pulido", {}),
        "historial_afinados": snapshot.get("historial_afinados", []),
        "imagen": imagen,
        "audio_bytes": audio_bytes,
        "audio_url": snapshot.get("audio_url", metadata.get("audio_url")),
        "audio_status": snapshot.get("audio_status", metadata.get("audio_status")),
        "audio_task_id": snapshot.get("audio_task_id", metadata.get("audio_task_id")),
        "audio_estilo": snapshot.get("audio_estilo", ""),
        "audio_error": snapshot.get("audio_error"),
    }

    return recuperado, metadata

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
    if "audio_style_selected" not in st.session_state:
        st.session_state.audio_style_selected = ""
    if "aviso_guardado_legacy" not in st.session_state:
        st.session_state.aviso_guardado_legacy = ""

    st.title("Generador de Poesía V2: Sindar")
    st.markdown("UI organizada en dos partes: izquierda para configuración/guardado y derecha para poema, música e imagen.")

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.subheader("Modelos de IA")
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

        st.markdown("---")
        st.subheader("Guardar y recuperar")
        resultado_actual = st.session_state.ultimo_resultado
        nombre_guardado = st.text_input("Nombre del guardado", value="poema", key="nombre_guardado")

        if st.button("Grabar", use_container_width=True, disabled=not bool(resultado_actual.get("poema_final"))):
            try:
                bundle_id = _guardar_bundle(
                    resultado=resultado_actual,
                    params=st.session_state.ultimo_params,
                    nombre=nombre_guardado
                )
                st.session_state.ultimo_bundle_id = bundle_id
                st.session_state.aviso_guardado_legacy = ""
                st.success(f"Guardado creado: {bundle_id}")
            except Exception as e:
                st.error(f"Error al grabar: {e}")

        opciones = _listar_guardados()
        if opciones:
            seleccionado = st.selectbox("Guardados disponibles", options=opciones, index=0, key="guardado_seleccionado")
        else:
            seleccionado = None
            st.caption("No hay guardados todavía.")

        if st.button("Recuperar", use_container_width=True, disabled=not bool(opciones)):
            try:
                recuperado, meta = _recuperar_bundle(seleccionado)
                st.session_state.ultimo_resultado = recuperado
                st.session_state.ultimo_params = meta.get("params", st.session_state.ultimo_params)
                st.session_state.ultimo_bundle_id = seleccionado
                st.session_state.audio_style_selected = recuperado.get("audio_estilo", "") or ""
                st.session_state.estilo_musica_input = st.session_state.audio_style_selected
                if meta.get("es_legacy"):
                    st.session_state.aviso_guardado_legacy = (
                        "Guardado legacy recuperado: puede no incluir detalles completos del proceso "
                        "(poema inicial, crítica y poema corregido)."
                    )
                else:
                    st.session_state.aviso_guardado_legacy = ""
                st.success(f"Guardado recuperado: {seleccionado}")
                if meta.get("audio_url"):
                    st.caption("Incluye URL de audio original y copia local si estaba disponible.")
                st.rerun()
            except Exception as e:
                st.error(f"Error al recuperar: {e}")

        if st.session_state.ultimo_bundle_id:
            st.caption(f"Último guardado usado: {st.session_state.ultimo_bundle_id}")

    with col_right:
        st.subheader("Poema")
        if st.session_state.aviso_guardado_legacy:
            st.warning(st.session_state.aviso_guardado_legacy)
        col_poema_a, col_poema_b = st.columns(2)
        with col_poema_a:
            estilo = st.text_input("Estilo", value="Estilo libre pero lírico")
            extension = st.selectbox("Extensión", ["muy corta", "corta", "media", "larga"], index=2)
        with col_poema_b:
            tema = st.text_input("Tema", value="La emoción del básquet")
            tono_extra = st.text_input("Tono Extra", value="Épico y apasionado")

        restricciones = st.text_area(
            "Restricciones",
            value="Sin rima consonante forzada, sin referencias tecnológicas"
        )

        if st.button("Generar poema", type="primary", use_container_width=True):
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
                    "crear_imagen": False,
                    "crear_audio": False,
                }

                with st.spinner("🤖 El agente está consultando la obra, escribiendo y puliendo..."):
                    try:
                        resultado = ejecutar_pipeline_poetico(params)
                        st.session_state.ultimo_resultado = resultado
                        st.session_state.ultimo_params = params
                        st.session_state.audio_style_selected = ""
                        st.session_state.aviso_guardado_legacy = ""
                        st.success("¡Poema generado con éxito!")
                    except Exception as e:
                        st.error(f"❌ Ocurrió un error: {e}")

        resultado_actual = st.session_state.ultimo_resultado
        if resultado_actual.get("poema_final"):
            st.text_area("Resultado del poema", value=resultado_actual.get("poema_final", ""), height=320)

            comentario_rework = st.text_area(
                "Comentario para afinar",
                value="",
                key="comentario_rework",
                placeholder="Ejemplo: hazlo más íntimo, reduce metáforas bélicas y cierra con una imagen esperanzadora"
            )

            if st.button("Afinar Poema", use_container_width=True):
                with st.spinner("Afinando poema con tu comentario (rework -> evaluación -> reescritura -> pulido)..."):
                    try:
                        params_ultimo = st.session_state.ultimo_params or {}
                        insumos_pulido = resultado_actual.get("insumos_pulido", {}) or {}
                        contexto_para_pulido = (
                            resultado_actual.get("contexto_extendido")
                            or insumos_pulido.get("contexto_extendido")
                            or ""
                        )
                        params_rework = {
                            "poema_base": resultado_actual.get("poema_final", ""),
                            "comentario_rework": comentario_rework,
                            "tema": params_ultimo.get("tema", ""),
                            "tono_extra": params_ultimo.get("tono_extra", ""),
                            "groq_model": groq_model,
                            "google_model": google_model,
                            "contexto_extendido": contexto_para_pulido,
                        }
                        refinado = ejecutar_rework_poetico(params_rework)

                        # Sustituimos el resultado textual y limpiamos artefactos para evitar desalineación.
                        nuevo_resultado = dict(st.session_state.ultimo_resultado)
                        historial = list(nuevo_resultado.get("historial_afinados", []))
                        historial.append({
                            "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
                            "comentario": comentario_rework,
                            "poema_entrada": resultado_actual.get("poema_final", ""),
                            "poema_salida": refinado.get("poema_final", ""),
                        })
                        nuevo_resultado.update(refinado)
                        nuevo_resultado["historial_afinados"] = historial
                        nuevo_resultado["imagen"] = None
                        nuevo_resultado["audio_url"] = None
                        nuevo_resultado["audio_bytes"] = None
                        nuevo_resultado["audio_task_id"] = None
                        nuevo_resultado["audio_status"] = None
                        nuevo_resultado["audio_error"] = None
                        nuevo_resultado["audio_estilo"] = None
                        st.session_state.ultimo_resultado = nuevo_resultado
                        st.session_state.audio_style_selected = ""
                        st.success("Poema afinado correctamente.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error afinando poema: {e}")

            historial_afinados = resultado_actual.get("historial_afinados", [])
            if historial_afinados:
                with st.expander(f"Historial de afinados ({len(historial_afinados)})", expanded=False):
                    for idx, item in enumerate(reversed(historial_afinados), start=1):
                        st.markdown(f"**Afinado {idx} · {item.get('fecha', 'sin fecha')}**")
                        st.caption(item.get("comentario", "Sin comentario"))
                        poema_vista = item.get("poema_salida") or ""
                        st.text_area(
                            f"Salida {idx}",
                            value=poema_vista,
                            height=120,
                            key=f"historial_afinado_salida_{idx}",
                        )
        else:
            st.info("Genera o recupera un poema para continuar con música e imagen.")

        st.markdown("---")
        st.subheader("Música (Lyria 3)")
        estilo_musica_input = st.text_input(
            "Estilo de la canción (opcional)",
            value=st.session_state.audio_style_selected,
            key="estilo_musica_input"
        )

        if st.button("Usar este estilo / generar estilo automáticamente", use_container_width=True):
            if not resultado_actual.get("poema_final"):
                st.warning("Primero genera o recupera un poema.")
            else:
                try:
                    if (estilo_musica_input or "").strip():
                        st.session_state.audio_style_selected = estilo_musica_input.strip()
                    else:
                        params_ultimo = st.session_state.ultimo_params or {}
                        contexto_audio = resultado_actual.get("contexto_extendido") or (
                            f"TEMA: {params_ultimo.get('tema', '')}\nTONO: {params_ultimo.get('tono_extra', '')}"
                        )
                        st.session_state.audio_style_selected = generar_estilo_musical_desde_contexto(
                            contexto_poetico=contexto_audio,
                            poema_texto=resultado_actual.get("poema_final", ""),
                            tema=params_ultimo.get("tema", ""),
                            tono_extra=params_ultimo.get("tono_extra", ""),
                            model=google_model,
                        )
                    st.success("Estilo musical listo.")
                except Exception as e:
                    st.error(f"Error preparando estilo musical: {e}")

        if st.session_state.audio_style_selected:
            st.caption(f"Estilo actual: {st.session_state.audio_style_selected}")

        with st.expander("Opciones avanzadas de audio", expanded=False):
            audio_title = st.text_input("Título de la canción (opcional)", value="", key="audio_title")
            audio_instrumental = st.checkbox("Modo instrumental", value=False, key="audio_instrumental")
            audio_wait = st.checkbox("Esperar audio final en la respuesta", value=True, key="audio_wait")
            audio_duracion_seg = st.slider(
                "Duración objetivo (segundos)",
                min_value=30,
                max_value=240,
                value=90,
                step=10,
                key="audio_duracion_seg"
            )

        if st.button("Generar canción", use_container_width=True, disabled=not bool(resultado_actual.get("poema_final"))):
            with st.spinner("Generando canción desde el poema actual..."):
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
                        model_google=google_model,
                        estilo_musical_override=st.session_state.audio_style_selected,
                        titulo=audio_title,
                        instrumental=audio_instrumental,
                        wait_audio=audio_wait,
                        duracion_segundos=audio_duracion_seg,
                    )
                    st.session_state.ultimo_resultado["audio_url"] = audio_resultado.get("audio_url")
                    st.session_state.ultimo_resultado["audio_bytes"] = audio_resultado.get("audio_bytes")
                    st.session_state.ultimo_resultado["audio_estilo"] = audio_resultado.get("estilo_musical")
                    st.session_state.ultimo_resultado["audio_task_id"] = audio_resultado.get("task_id")
                    st.session_state.ultimo_resultado["audio_status"] = audio_resultado.get("status")
                    st.session_state.ultimo_resultado["audio_error"] = None
                    st.session_state.audio_style_selected = audio_resultado.get("estilo_musical", "")
                    st.success("Canción solicitada correctamente.")
                except Exception as e:
                    st.session_state.ultimo_resultado["audio_error"] = str(e)
                    st.error(f"Error generando canción: {e}")

        if resultado_actual.get("audio_url"):
            st.audio(resultado_actual["audio_url"], format="audio/mpeg")
            st.markdown(f"[Abrir audio en nueva pestaña]({resultado_actual['audio_url']})")
        elif resultado_actual.get("audio_bytes"):
            st.audio(resultado_actual["audio_bytes"], format="audio/mpeg")
        elif resultado_actual.get("audio_task_id"):
            st.info(
                f"Audio en proceso. Task ID: {resultado_actual.get('audio_task_id')} | "
                f"Estado: {resultado_actual.get('audio_status', 'processing')}"
            )
        elif resultado_actual.get("audio_error"):
            st.warning(f"No se pudo generar audio: {resultado_actual['audio_error']}")

        with st.expander("Estado de audio", expanded=False):
            st.caption("Con Google AI la generación de audio es directa y no usa taskId para polling.")
            if resultado_actual.get("audio_status"):
                st.write(f"Estado: {resultado_actual.get('audio_status')}")

        st.markdown("---")
        st.subheader("Imagen")
        if resultado_actual.get("imagen"):
            st.image(resultado_actual["imagen"], caption="Imagen generada a partir del poema.")
        else:
            st.caption("Aún no hay imagen generada para este poema.")

        if st.button("Generar imagen", use_container_width=True, disabled=not bool(resultado_actual.get("poema_final"))):
            with st.spinner("Generando imagen desde el poema actual..."):
                try:
                    nueva_imagen = generate_from_poem(resultado_actual.get("poema_final", ""))
                    st.session_state.ultimo_resultado["imagen"] = nueva_imagen
                    st.success("Imagen generada correctamente.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generando imagen: {e}")

        with st.expander("Ver detalles del proceso"):
            st.markdown("**1. Poema Inicial (Gemini + RAG):**")
            st.text(resultado_actual.get("poema_inicial", ""))
            st.markdown("**2. Crítica (Groq):**")
            st.json(resultado_actual.get("critica_final", {}))
            st.markdown("**3. Poema Corregido:**")
            st.text(resultado_actual.get("poema_corregido", ""))
            if resultado_actual.get("audio_estilo"):
                st.markdown("**4. Estilo musical usado:**")
                st.text(resultado_actual.get("audio_estilo", ""))

if __name__ == "__main__":
    main()