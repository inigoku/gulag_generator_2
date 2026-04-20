import os
import chromadb
from chromadb.utils import embedding_functions

def crear_chroma(ruta):
    os.makedirs(ruta, exist_ok=True)
    client = chromadb.PersistentClient(path=ruta)
    nombre_coleccion = os.path.basename(os.path.normpath(ruta))
    return client.get_or_create_collection(name=nombre_coleccion)

def abrir_chroma(ruta):
    os.makedirs(ruta, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=ruta)
        nombre_coleccion = os.path.basename(os.path.normpath(ruta))
        return client.get_collection(name=nombre_coleccion)
    except Exception as e:
        print(f"Advertencia: No se pudo abrir la base de datos Chroma en {ruta}: {e}")
        return None

def generar_embeddings(chunks):
    # Usamos el modelo por defecto de Chroma (all-MiniLM-L6-v2)
    ef = embedding_functions.DefaultEmbeddingFunction()
    return ef(chunks)

def insertar_en_chroma(collection, chunks, embeddings):
    ids = [str(i) for i in range(len(chunks))]
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=chunks[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

def buscar_en_chroma(collection, query, k=30):
    if collection is None:
        return []
    resultados = collection.query(
        query_texts=[query],
        n_results=k
    )
    if resultados and resultados['documents']:
        return resultados['documents'][0]
    return []