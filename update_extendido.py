import os
from generar_datos_iniciales import (
    extraer_texto_de_pdfs, limpiar_y_normalizar, guardar_texto, trocear_texto, 
    dividir_en_cinco, guardar_json, crear_chroma, generar_embeddings, 
    insertar_en_chroma, buscar_en_chroma, deepseek_analizar_estilo, 
    mezclar_cinco_perfiles, mezclar_perfiles
)

def procesar_extendido():
    print("Iniciando procesamiento de extendido (en 5 partes)...")
    rutas = {
        "pdfs_extendido": "./data/pdfs/extendido/",
        "corpus_extendido": "./data/corpus/extendido.txt",
        "chunks_extendido1": "./data/chunks/chunks_extendido1.json",
        "chunks_extendido2": "./data/chunks/chunks_extendido2.json",
        "chunks_extendido3": "./data/chunks/chunks_extendido3.json",
        "chunks_extendido4": "./data/chunks/chunks_extendido4.json",
        "chunks_extendido5": "./data/chunks/chunks_extendido5.json",
        "chroma_extendido1": "./data/chroma/extendido1/",
        "chroma_extendido2": "./data/chroma/extendido2/",
        "chroma_extendido3": "./data/chroma/extendido3/",
        "chroma_extendido4": "./data/chroma/extendido4/",
        "chroma_extendido5": "./data/chroma/extendido5/"
    }

    # Extraer texto extendido
    texto_extendido = extraer_texto_de_pdfs(rutas["pdfs_extendido"])
    texto_extendido = limpiar_y_normalizar(texto_extendido)
    guardar_texto(texto_extendido, rutas["corpus_extendido"])
    print("Texto extendido guardado.")

    # Trocear semántico
    chunks_extendido = trocear_texto(texto_extendido, tamaño=400)
    chunks_extendido1, chunks_extendido2, chunks_extendido3, chunks_extendido4, chunks_extendido5 = dividir_en_cinco(chunks_extendido)
    guardar_json(chunks_extendido1, rutas["chunks_extendido1"])
    guardar_json(chunks_extendido2, rutas["chunks_extendido2"])
    guardar_json(chunks_extendido3, rutas["chunks_extendido3"])
    guardar_json(chunks_extendido4, rutas["chunks_extendido4"])
    guardar_json(chunks_extendido5, rutas["chunks_extendido5"])
    print("Chunks extendidos guardados (5 partes).")

    # Embeddings y Chroma
    chroma_extendido1 = crear_chroma(rutas["chroma_extendido1"])
    chroma_extendido2 = crear_chroma(rutas["chroma_extendido2"])
    chroma_extendido3 = crear_chroma(rutas["chroma_extendido3"])
    chroma_extendido4 = crear_chroma(rutas["chroma_extendido4"])
    chroma_extendido5 = crear_chroma(rutas["chroma_extendido5"])
    
    embeddings_extendido1 = generar_embeddings(chunks_extendido1)
    embeddings_extendido2 = generar_embeddings(chunks_extendido2)
    embeddings_extendido3 = generar_embeddings(chunks_extendido3)
    embeddings_extendido4 = generar_embeddings(chunks_extendido4)
    embeddings_extendido5 = generar_embeddings(chunks_extendido5)

    insertar_en_chroma(chroma_extendido1, chunks_extendido1, embeddings_extendido1)
    insertar_en_chroma(chroma_extendido2, chunks_extendido2, embeddings_extendido2)
    insertar_en_chroma(chroma_extendido3, chunks_extendido3, embeddings_extendido3)
    insertar_en_chroma(chroma_extendido4, chunks_extendido4, embeddings_extendido4)
    insertar_en_chroma(chroma_extendido5, chunks_extendido5, embeddings_extendido5)
    print("Chroma extendidos generados (5 partes).")

    # Auto-afinación estilística extendida
    tema = "análisis de estilo"
    contexto_extendido1 = buscar_en_chroma(chroma_extendido1, tema, k=15)
    contexto_extendido2 = buscar_en_chroma(chroma_extendido2, tema, k=15)
    contexto_extendido3 = buscar_en_chroma(chroma_extendido3, tema, k=15)
    contexto_extendido4 = buscar_en_chroma(chroma_extendido4, tema, k=15)
    contexto_extendido5 = buscar_en_chroma(chroma_extendido5, tema, k=15)

    print("Llamando a DeepSeek para analizar extendido 1 a 5...")
    perfil_extendido1 = deepseek_analizar_estilo(contexto_extendido1)
    perfil_extendido2 = deepseek_analizar_estilo(contexto_extendido2)
    perfil_extendido3 = deepseek_analizar_estilo(contexto_extendido3)
    perfil_extendido4 = deepseek_analizar_estilo(contexto_extendido4)
    perfil_extendido5 = deepseek_analizar_estilo(contexto_extendido5)
    
    print("Mezclando 5 perfiles...")
    perfil_extendido = mezclar_cinco_perfiles(perfil_extendido1, perfil_extendido2, perfil_extendido3, perfil_extendido4, perfil_extendido5)

    guardar_texto(perfil_extendido1, "./estilo/perfil_extendido1.md")
    guardar_texto(perfil_extendido2, "./estilo/perfil_extendido2.md")
    guardar_texto(perfil_extendido3, "./estilo/perfil_extendido3.md")
    guardar_texto(perfil_extendido4, "./estilo/perfil_extendido4.md")
    guardar_texto(perfil_extendido5, "./estilo/perfil_extendido5.md")
    guardar_texto(perfil_extendido, "./estilo/perfil_extendido.md")

    # Mezcla final de estilos con los ya existentes y nuevos pesos
    print("Mezclando todos los perfiles...")
    with open("./estilo/perfil_obra.md", "r", encoding="utf-8") as f:
        perfil_obra = f.read()
    with open("./estilo/perfil_influencias.md", "r", encoding="utf-8") as f:
        perfil_influencias = f.read()
    with open("./estilo/perfil_folklore.md", "r", encoding="utf-8") as f:
        perfil_folklore = f.read()

    alpha, beta, gamma, delta = 0.5, 0.25, 0.15, 0.10

    perfil_estilistico_final = mezclar_perfiles(
        perfil_obra, perfil_influencias, perfil_folklore, perfil_extendido, 
        alpha, beta, gamma, delta
    )

    guardar_texto(perfil_estilistico_final, "./estilo/perfil_estilistico_final.md")
    guardar_json({"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta}, "./estilo/mezcla_estilo.json")
    print("Procesamiento completado y perfil estilístico final actualizado.")

if __name__ == "__main__":
    procesar_extendido()