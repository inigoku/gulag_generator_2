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

from chroma import crear_chroma, generar_embeddings, insertar_en_chroma, buscar_en_chroma

from config import (
    GROQ_API_KEY, GROQ_MODEL, REWORK_RETRIES,
    GOOGLE_MODEL, GOOGLE_API_KEY,
    BRAVE_SEARCH_API_KEY, DEEPSEEK_API_KEY, DEEPSEEK_MODEL
)

NUM_EXTENDIDOS = 10

def cargar_configuracion(env_path, modelos_path, pesos_path):
    if os.path.exists(env_path):
        load_dotenv(env_path)
    
    config = {"pesos_estilo": {"obra": 0.5, "influencias": 0.25, "folklore": 0.15, "extendido": 0.10}}
    
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

def dividir_en_dos(chunks):
    mitad = len(chunks) // 2
    return chunks[:mitad], chunks[mitad:]

def dividir_en_n(chunks, n):
    if n <= 1:
        return [chunks]
    k = len(chunks) // n
    partes = [chunks[i * k:(i + 1) * k] for i in range(n - 1)]
    partes.append(chunks[(n - 1) * k:])
    return partes

def guardar_json(data, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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
    ruta_prompt = "./prompts/prompt_analisis_estilo.txt"
    if os.path.exists(ruta_prompt):
        with open(ruta_prompt, 'r', encoding='utf-8') as f:
            instrucciones = f.read().strip()
    else:
        instrucciones = "Analiza el estilo literario de los siguientes textos:"

    contexto = "\n\n".join(fragmentos[:15]) 
    prompt = f"{instrucciones}\n\n{contexto}"
    return _llamar_deepseek(prompt)

def mezclar_dos_perfiles(perfil1, perfil2):
    prompt = (
        "Mezcla estos dos perfiles al 50% y devuelve un único perfil estilístico coherente.\n\n"
        f"[P1]\n{perfil1}\n\n[P2]\n{perfil2}"
    )
    return _llamar_deepseek(prompt)

def mezclar_n_perfiles(perfiles):
    bloques = "\n\n".join([f"[P{i}]\n{perfil}" for i, perfil in enumerate(perfiles, start=1)])
    prompt = (
        f"Mezcla estos {len(perfiles)} perfiles equitativamente y devuelve un único perfil estilístico coherente.\n\n"
        f"{bloques}"
    )
    return _llamar_deepseek(prompt)

def mezclar_perfiles(perfil1, perfil2, perfil3, perfil4, alpha, beta, gamma, delta):
    prompt = f"Mezcla estos perfiles con pesos: P1({alpha}), P2({beta}), P3({gamma}), P4({delta}).\n\n[P1]\n{perfil1}\n\n[P2]\n{perfil2}\n\n[P3]\n{perfil3}\n\n[P4]\n{perfil4}"
    return _llamar_deepseek(prompt)

def generar_datos_iniciales():

    ###############################################
    # 1. CONFIGURACIÓN
    ###############################################

    config = cargar_configuracion("./config/claves.env",
                                  "./config/modelos.yaml",
                                  "./config/pesos_estilo.yaml")

    rutas = {
        "pdfs_obra": "./data/pdfs/obra/",
        "pdfs_influencias": "./data/pdfs/influencias/",
        "pdfs_extendido": "./data/pdfs/extendido/",
        "corpus_obra": "./data/corpus/obra.txt",
        "corpus_influencias": "./data/corpus/influencias.txt",
        "corpus_folklore": "./data/corpus/folklore.txt",
        "corpus_extendido": "./data/corpus/extendido.txt",
        "chunks_obra": "./data/chunks/chunks_obra.json",
        "chunks_influencias": "./data/chunks/chunks_influencias.json",
        "chunks_folklore1": "./data/chunks/chunks_folklore1.json",
        "chunks_folklore2": "./data/chunks/chunks_folklore2.json",
        "chroma_obra": "./data/chroma/obra/",
        "chroma_influencias": "./data/chroma/influencias/",
        "chroma_folklore1": "./data/chroma/folklore1/",
        "chroma_folklore2": "./data/chroma/folklore2/"
    }

    for i in range(1, NUM_EXTENDIDOS + 1):
        rutas[f"chunks_extendido{i}"] = f"./data/chunks/chunks_extendido{i}.json"
        rutas[f"chroma_extendido{i}"] = f"./data/chroma/extendido{i}/"

    # Crear directorios necesarios si no existen
    for ruta in rutas.values():
        # Si tiene extensión es archivo (usar dirname), si no es directorio
        directorio = os.path.dirname(ruta) if os.path.splitext(ruta)[1] else ruta
        if directorio and not os.path.exists(directorio):
            os.makedirs(directorio, exist_ok=True)
            print(f"Creado directorio: {directorio}")


    ###############################################
    # 2. CARGA DE CORPUS
    ###############################################

    texto_obra = extraer_texto_de_pdfs(rutas["pdfs_obra"])
    texto_influencias = extraer_texto_de_pdfs(rutas["pdfs_influencias"])
    texto_extendido = extraer_texto_de_pdfs(rutas["pdfs_extendido"])

    # El corpus de folklore ya fue descargado a texto directamente
    if os.path.exists(rutas["corpus_folklore"]):
        with open(rutas["corpus_folklore"], 'r', encoding='utf-8') as f:
            texto_folklore = f.read()
    else:
        texto_folklore = ""
        print("⚠️ No se encontró el corpus de folklore.")

    texto_obra = limpiar_y_normalizar(texto_obra)
    texto_influencias = limpiar_y_normalizar(texto_influencias)
    texto_folklore = limpiar_y_normalizar(texto_folklore)
    texto_extendido = limpiar_y_normalizar(texto_extendido)

    guardar_texto(texto_obra, rutas["corpus_obra"])
    guardar_texto(texto_influencias, rutas["corpus_influencias"])
    guardar_texto(texto_folklore, rutas["corpus_folklore"])
    guardar_texto(texto_extendido, rutas["corpus_extendido"])


    ###############################################
    # 3. TROCEO SEMÁNTICO
    ###############################################

    chunks_obra = trocear_texto(texto_obra, tamaño=400)
    chunks_influencias = trocear_texto(texto_influencias, tamaño=400)
    chunks_folklore = trocear_texto(texto_folklore, tamaño=400)
    chunks_folklore1, chunks_folklore2 = dividir_en_dos(chunks_folklore)
    chunks_extendido = trocear_texto(texto_extendido, tamaño=400)
    chunks_extendidos = dividir_en_n(chunks_extendido, NUM_EXTENDIDOS)

    guardar_json(chunks_obra, rutas["chunks_obra"])
    guardar_json(chunks_influencias, rutas["chunks_influencias"])
    guardar_json(chunks_folklore1, rutas["chunks_folklore1"])
    guardar_json(chunks_folklore2, rutas["chunks_folklore2"])
    for i, chunks_ext in enumerate(chunks_extendidos, start=1):
        guardar_json(chunks_ext, rutas[f"chunks_extendido{i}"])


    ###############################################
    # 4. EMBEDDINGS + CHROMADB
    ###############################################

    chroma_obra = crear_chroma(rutas["chroma_obra"])
    chroma_influencias = crear_chroma(rutas["chroma_influencias"])
    chroma_folklore1 = crear_chroma(rutas["chroma_folklore1"])
    chroma_folklore2 = crear_chroma(rutas["chroma_folklore2"])
    chromas_extendido = [
        crear_chroma(rutas[f"chroma_extendido{i}"])
        for i in range(1, NUM_EXTENDIDOS + 1)
    ]

    embeddings_obra = generar_embeddings(chunks_obra)
    embeddings_influencias = generar_embeddings(chunks_influencias)
    embeddings_folklore1 = generar_embeddings(chunks_folklore1)
    embeddings_folklore2 = generar_embeddings(chunks_folklore2)
    embeddings_extendidos = [
        generar_embeddings(chunks_ext)
        for chunks_ext in chunks_extendidos
    ]

    insertar_en_chroma(chroma_obra, chunks_obra, embeddings_obra)
    insertar_en_chroma(chroma_influencias, chunks_influencias, embeddings_influencias)
    insertar_en_chroma(chroma_folklore1, chunks_folklore1, embeddings_folklore1)
    insertar_en_chroma(chroma_folklore2, chunks_folklore2, embeddings_folklore2)
    for chroma_ext, chunks_ext, embeddings_ext in zip(chromas_extendido, chunks_extendidos, embeddings_extendidos):
        insertar_en_chroma(chroma_ext, chunks_ext, embeddings_ext)


    ###############################################
    # 5. RECUPERACIÓN
    ###############################################

    tema = "análisis de estilo"   # tema neutro para extraer patrones

    contexto_obra = buscar_en_chroma(chroma_obra, tema, k=30)
    contexto_influencias = buscar_en_chroma(chroma_influencias, tema, k=30)
    contexto_folklore1 = buscar_en_chroma(chroma_folklore1, tema, k=15)
    contexto_folklore2 = buscar_en_chroma(chroma_folklore2, tema, k=15)
    contexto_folklore = contexto_folklore1 + contexto_folklore2
    contexto_extendidos = [buscar_en_chroma(chroma_ext, tema, k=15) for chroma_ext in chromas_extendido]
    contexto_extendido_text = sum(contexto_extendidos, [])


    ###############################################
    # 5bis. AUTO-AFINACIÓN ESTILÍSTICA (DEEPSEEK)
    ###############################################

    # 5bis.1 análisis de estilo de cada corpus
    perfil_obra = deepseek_analizar_estilo(contexto_obra)
    perfil_influencias = deepseek_analizar_estilo(contexto_influencias)
    perfil_folklore1 = deepseek_analizar_estilo(contexto_folklore1)
    perfil_folklore2 = deepseek_analizar_estilo(contexto_folklore2)
    perfil_folklore = mezclar_dos_perfiles(perfil_folklore1, perfil_folklore2)
    perfiles_extendidos = [
        deepseek_analizar_estilo(contexto_extendido)
        for contexto_extendido in contexto_extendidos
    ]
    perfil_extendido = mezclar_n_perfiles(perfiles_extendidos)

    guardar_texto(perfil_obra, "./estilo/perfil_obra.md")
    guardar_texto(perfil_influencias, "./estilo/perfil_influencias.md")
    guardar_texto(perfil_folklore1, "./estilo/perfil_folklore1.md")
    guardar_texto(perfil_folklore2, "./estilo/perfil_folklore2.md")
    guardar_texto(perfil_folklore, "./estilo/perfil_folklore.md")
    for i, perfil_ext in enumerate(perfiles_extendidos, start=1):
        guardar_texto(perfil_ext, f"./estilo/perfil_extendido{i}.md")
    guardar_texto(perfil_extendido, "./estilo/perfil_extendido.md")

    # 5bis.2 mezcla ponderada
    pesos = config.get("pesos_estilo", {})
    α = pesos.get("obra", 0.5)
    β = pesos.get("influencias", 0.25)
    γ = pesos.get("folklore", 0.15)
    δ = pesos.get("extendido", 0.10)

    perfil_estilistico_final = mezclar_perfiles(perfil_obra, perfil_influencias, perfil_folklore, perfil_extendido, α, β, γ, δ)

    guardar_texto(perfil_estilistico_final, "./estilo/perfil_estilistico_final.md")
    guardar_json({"alpha": α, "beta": β, "gamma": γ, "delta": δ}, "./estilo/mezcla_estilo.json")



    ###############################################
    # FIN
    ###############################################

    return {
        "perfil_estilistico": perfil_estilistico_final,
        "contexto_obra": contexto_obra,
        "contexto_influencias": contexto_influencias,
        "contexto_folklore": contexto_folklore,
        "contexto_extendido": contexto_extendido_text
    }

if __name__ == "__main__":
    print("=== Iniciando generación de datos iniciales ===")
    generar_datos_iniciales()
