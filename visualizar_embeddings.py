import chromadb
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import os
import argparse

def cargar_embeddings_de_coleccion(path_coleccion):
    """
    Carga embeddings y documentos de una colección de ChromaDB.
    """
    if not os.path.exists(path_coleccion):
        print(f"Advertencia: La ruta de la colección no existe: {path_coleccion}")
        return None, None

    try:
        client = chromadb.PersistentClient(path=path_coleccion)
        # Asumimos que solo hay una colección por directorio, así que obtenemos la primera.
        collections = client.list_collections()
        if not collections:
            print(f"No se encontraron colecciones en {path_coleccion}")
            return None, None
        
        collection = client.get_collection(name=collections[0].name)
        
        # Obtenemos embeddings y los documentos para usarlos como texto en el hover
        data = collection.get(include=["embeddings", "documents"])
        
        hover_texts = data.get('documents')
        embeddings = np.array(data.get('embeddings'))

        if hover_texts is None or embeddings is None:
            return None, None

        return embeddings, hover_texts
    except Exception as e:
        print(f"Error al cargar la colección desde {path_coleccion}: {e}")
        return None, None

def main():
    """
    Función principal para visualizar los embeddings.
    """
    parser = argparse.ArgumentParser(description="Visualizar embeddings en 3D")
    parser.add_argument("--metodo", choices=["pca", "umap"], default="pca", help="Método de reducción (pca o umap)")
    parser.add_argument("--output", type=str, default=None, help="Guardar gráfico como HTML (ej: grafico.html)")
    args = parser.parse_args()

    base_path = "." # Asume que se ejecuta desde generador_v2/
    path_obra = os.path.join(base_path, "data/chroma/obra")
    path_influencias = os.path.join(base_path, "data/chroma/influencias")
    path_folklore = os.path.join(base_path, "data/chroma/folklore")

    # Diagnosticar rutas
    print(f"📍 Buscando datos en: {os.path.abspath(base_path)}")
    print(f"   - Obra: {os.path.abspath(path_obra)} {'✓' if os.path.exists(path_obra) else '✗ NO EXISTE'}")
    print(f"   - Influencias: {os.path.abspath(path_influencias)} {'✓' if os.path.exists(path_influencias) else '✗ NO EXISTE'}")
    print(f"   - Folklore: {os.path.abspath(path_folklore)} {'✓' if os.path.exists(path_folklore) else '✗ NO EXISTE'}")
    print()

    # Cargar embeddings
    print("Cargando embeddings de 'obra'...")
    embeddings_obra, hover_obra = cargar_embeddings_de_coleccion(path_obra)
    if embeddings_obra is not None:
        print(f"  ✓ {len(embeddings_obra)} embeddings cargados")
    
    print("Cargando embeddings de 'influencias'...")
    embeddings_influencias, hover_influencias = cargar_embeddings_de_coleccion(path_influencias)
    if embeddings_influencias is not None:
        print(f"  ✓ {len(embeddings_influencias)} embeddings cargados")

    print("Cargando embeddings de 'folklore'...")
    embeddings_folklore, hover_folklore = cargar_embeddings_de_coleccion(path_folklore)
    if embeddings_folklore is not None:
        print(f"  ✓ {len(embeddings_folklore)} embeddings cargados")

    if embeddings_obra is None and embeddings_influencias is None and embeddings_folklore is None:
        print("No se pudieron cargar embeddings de ninguna colección. Saliendo.")
        return

    # Combinar embeddings y crear etiquetas
    all_embeddings = []
    labels = []
    hover_texts = []

    if embeddings_obra is not None:
        all_embeddings.append(embeddings_obra)
        labels.extend(['Obra'] * len(embeddings_obra))
        hover_texts.extend(hover_obra)

    if embeddings_influencias is not None:
        all_embeddings.append(embeddings_influencias)
        labels.extend(['Influencias'] * len(embeddings_influencias))
        hover_texts.extend(hover_influencias)

    if embeddings_folklore is not None:
        all_embeddings.append(embeddings_folklore)
        labels.extend(['Folklore'] * len(embeddings_folklore))
        hover_texts.extend(hover_folklore)

    all_embeddings = np.vstack(all_embeddings)

    # Reducción de dimensionalidad a 3D
    if args.metodo == 'umap':
        try:
            import umap
        except ImportError:
            print("Error: Para usar UMAP necesitas instalar la librería.")
            print("Ejecuta: pip install umap-learn")
            return

        print(f"Reduciendo {all_embeddings.shape[0]} embeddings a 3D usando UMAP...")
        reducer = umap.UMAP(n_components=3, random_state=42)
        embeddings_3d = reducer.fit_transform(all_embeddings)
    else:
        print(f"Reduciendo {all_embeddings.shape[0]} embeddings a 3D usando PCA...")
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(all_embeddings)

    # Definir colores según etiqueta
    color_map = {'Obra': 0, 'Influencias': 0.5, 'Folklore': 1}
    colors = [color_map[label] for label in labels]

    # Crear la figura 3D con Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='markers',
        marker=dict(size=5, color=colors, colorscale='Viridis', opacity=0.8),
        text=hover_texts,
        customdata=labels,
        hovertemplate='<b>%{customdata}</b><br><br>%{text}<extra></extra>'
    )])

    fig.update_layout(title=f'Visualización 3D de Embeddings ({args.metodo.upper()} - Obra vs. Influencias vs. Folklore)',
                      scene=dict(xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3'),
                      margin=dict(r=20, b=10, l=10, t=40))

    # Guardar o mostrar
    if args.output:
        output_path = os.path.join(base_path, args.output)
        fig.write_html(output_path)
        print(f"\n✓ Gráfico guardado en: {os.path.abspath(output_path)}")
        print(f"  Abre este archivo en tu navegador para verlo.")
    else:
        print("\nMostrando gráfico interactivo. Se abrirá una ventana en tu navegador.")
        print("Si no se abre, usa: python visualizar_embeddings.py --output grafico.html")
        print("Puedes rotar, hacer zoom y pasar el ratón sobre los puntos para ver el texto del chunk.")
        fig.show()

if __name__ == "__main__":
    main()