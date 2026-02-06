import os
import glob
import json
import yaml
import unicodedata
import requests
import pypdf
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from config import (
    GROQ_API_KEY, GROQ_MODEL, REWORK_RETRIES,
    GOOGLE_MODEL, GOOGLE_API_KEY,
    BRAVE_SEARCH_API_KEY, DEEPSEEK_API_KEY, DEEPSEEK_MODEL
)

def cargar_configuracion(env_path, modelos_path, pesos_path):
    if os.path.exists(env_path):
        load_dotenv(env_path)
    
    config = {"pesos_estilo": {"obra": 0.5, "influencias": 0.5}}
    
    if os.path.exists(modelos_path):
        with open(modelos_path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f)
            if loaded: config["modelos"] = loaded
            
    if os.path.exists(pesos_path):
        with open(pesos_path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f)
            if loaded: config["pesos_estilo"] = loaded
            
    return config

def extraer_texto_de_pdfs(ruta_carpeta):
    texto_acumulado = []
    patron = os.path.join(ruta_carpeta, "*.pdf")
    archivos = glob.glob(patron)
    
    for archivo in archivos:
        try:
            print(f"Extrayendo texto de: {archivo}")
            reader = pypdf.PdfReader(archivo)
            for page in reader.pages:
                t = page.extract_text()
                if t: texto_acumulado.append(t)
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")
            
    return "\n".join(texto_acumulado)

def limpiar_y_normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    texto = " ".join(texto.split())
    return texto

def guardar_texto(texto, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write(texto)

def trocear_texto(texto, tamaño=400):
    palabras = texto.split()
    chunks = []
    for i in range(0, len(palabras), tamaño):
        fragmento = " ".join(palabras[i:i+tamaño])
        chunks.append(fragmento)
    return chunks

def guardar_json(data, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def crear_chroma(ruta):
    os.makedirs(ruta, exist_ok=True)
    client = chromadb.PersistentClient(path=ruta)
    nombre_coleccion = os.path.basename(os.path.normpath(ruta))
    return client.get_or_create_collection(name=nombre_coleccion)

def generar_embeddings(chunks):
    # Usamos el modelo por defecto de Chroma (all-MiniLM-L6-v2)
    ef = embedding_functions.DefaultEmbeddingFunction()
    return ef(chunks)

def insertar_en_chroma(collection, chunks, embeddings):
    ids = [str(i) for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

def buscar_en_chroma(collection, query, k=30):
    resultados = collection.query(
        query_texts=[query],
        n_results=k
    )
    if resultados and resultados['documents']:
        return resultados['documents'][0]
    return []

def _llamar_deepseek(prompt):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️ DEEPSEEK_API_KEY no encontrada. Usando respuesta simulada.")
        return "Simulación: Análisis realizado."
        
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error API DeepSeek: {e}"

def deepseek_analizar_estilo(fragmentos):
    ruta_prompt = "./generador_v2/prompts/prompt_analisis_estilo.txt"
    if os.path.exists(ruta_prompt):
        with open(ruta_prompt, 'r', encoding='utf-8') as f:
            instrucciones = f.read().strip()
    else:
        instrucciones = "Analiza el estilo literario de los siguientes textos:"

    contexto = "\n\n".join(fragmentos[:15]) 
    prompt = f"{instrucciones}\n\n{contexto}"
    return _llamar_deepseek(prompt)

def mezclar_perfiles(perfil1, perfil2, alpha, beta):
    prompt = f"Mezcla estos perfiles con pesos: P1({alpha}), P2({beta}).\n\n[P1]\n{perfil1}\n\n[P2]\n{perfil2}"
    return _llamar_deepseek(prompt)

def generar_datos_iniciales():

    ###############################################
    # 1. CONFIGURACIÓN
    ###############################################

    config = cargar_configuracion("./generador_v2/config/claves.env",
                                  "./generador_v2/config/modelos.yaml",
                                  "./generador_v2/config/pesos_estilo.yaml")

    rutas = {
        "pdfs_obra": "./generador_v2/data/pdfs/obra/",
        "pdfs_influencias": "./generador_v2/data/pdfs/influencias/",
        "corpus_obra": "./generador_v2/data/corpus/obra.txt",
        "corpus_influencias": "./generador_v2/data/corpus/influencias.txt",
        "chunks_obra": "./generador_v2/data/chunks/chunks_obra.json",
        "chunks_influencias": "./generador_v2/data/chunks/chunks_influencias.json",
        "chroma_obra": "./generador_v2/data/chroma/obra/",
        "chroma_influencias": "./generador_v2/data/chroma/influencias/"
    }

    # Crear directorios necesarios si no existen
    for ruta in rutas.values():
        # Si tiene extensión es archivo (usar dirname), si no es directorio
        directorio = os.path.dirname(ruta) if os.path.splitext(ruta)[1] else ruta
        if directorio and not os.path.exists(directorio):
            os.makedirs(directorio, exist_ok=True)
            print(f"Creado directorio: {directorio}")


    ###############################################
    # 2. CARGA DE CORPUS (PDF → TEXTO)
    ###############################################

    texto_obra = extraer_texto_de_pdfs(rutas["pdfs_obra"])
    texto_influencias = extraer_texto_de_pdfs(rutas["pdfs_influencias"])

    texto_obra = limpiar_y_normalizar(texto_obra)
    texto_influencias = limpiar_y_normalizar(texto_influencias)

    guardar_texto(texto_obra, rutas["corpus_obra"])
    guardar_texto(texto_influencias, rutas["corpus_influencias"])



    ###############################################
    # 3. TROCEO SEMÁNTICO
    ###############################################

    chunks_obra = trocear_texto(texto_obra, tamaño=400)
    chunks_influencias = trocear_texto(texto_influencias, tamaño=400)

    guardar_json(chunks_obra, rutas["chunks_obra"])
    guardar_json(chunks_influencias, rutas["chunks_influencias"])



    ###############################################
    # 4. EMBEDDINGS + CHROMADB (DOBLE MEMORIA)
    ###############################################

    chroma_obra = crear_chroma(rutas["chroma_obra"])
    chroma_influencias = crear_chroma(rutas["chroma_influencias"])

    embeddings_obra = generar_embeddings(chunks_obra)
    embeddings_influencias = generar_embeddings(chunks_influencias)

    insertar_en_chroma(chroma_obra, chunks_obra, embeddings_obra)
    insertar_en_chroma(chroma_influencias, chunks_influencias, embeddings_influencias)



    ###############################################
    # 5. RECUPERACIÓN (RAG DUAL)
    ###############################################

    tema = "análisis de estilo"   # tema neutro para extraer patrones

    contexto_obra = buscar_en_chroma(chroma_obra, tema, k=30)
    contexto_influencias = buscar_en_chroma(chroma_influencias, tema, k=30)



    ###############################################
    # 5bis. AUTO-AFINACIÓN ESTILÍSTICA (DEEPSEEK)
    ###############################################

    # 5bis.1 análisis de estilo de cada corpus
    perfil_obra = deepseek_analizar_estilo(contexto_obra)
    perfil_influencias = deepseek_analizar_estilo(contexto_influencias)

    guardar_texto(perfil_obra, "./generador_v2/estilo/perfil_obra.md")
    guardar_texto(perfil_influencias, "./generador_v2/estilo/perfil_influencias.md")

    # 5bis.2 mezcla ponderada
    α = config["pesos_estilo"]["obra"]
    β = config["pesos_estilo"]["influencias"]

    perfil_estilistico_final = mezclar_perfiles(perfil_obra, perfil_influencias, α, β)

    guardar_texto(perfil_estilistico_final, "./generador_v2/estilo/perfil_estilistico_final.md")
    guardar_json({"alpha": α, "beta": β}, "./generador_v2/estilo/mezcla_estilo.json")



    ###############################################
    # FIN
    ###############################################

    return {
        "perfil_estilistico": perfil_estilistico_final,
        "contexto_obra": contexto_obra,
        "contexto_influencias": contexto_influencias
    }

if __name__ == "__main__":
    print("=== Iniciando generación de datos iniciales ===")
    generar_datos_iniciales()
