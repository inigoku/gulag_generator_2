def normalizar(pesos):
    total = sum(pesos.values())
    if total == 0:
        return pesos
    return {k: v / total for k, v in pesos.items()}

def calcular_pesos(perfil):
    """
    Calcula los pesos adaptativos del agente poético.
    """

    categoria = perfil.get("categoria")
    abstraccion = perfil.get("nivel_abstraccion")
    factualidad = perfil.get("grado_factualidad")
    densidad = perfil.get("densidad_metaforica")
    intencion = perfil.get("intencion_poetica")
    estilo_extra = perfil.get("estilo_extra", "")
    restricciones = perfil.get("restricciones", "")
    extension = perfil.get("extension", "")

    # Pesos base por categoría
    if categoria == "intimo":
        pesos = {"α": 0.45, "β": 0.25, "γ": 0.05, "δ": 0.15, "ε": 0.10}
    elif categoria == "conceptual":
        pesos = {"α": 0.40, "β": 0.30, "γ": 0.05, "δ": 0.15, "ε": 0.10}
    elif categoria == "experimental":
        pesos = {"α": 0.35, "β": 0.25, "γ": 0.05, "δ": 0.20, "ε": 0.15}
    else:
        pesos = {"α": 0.40, "β": 0.25, "γ": 0.10, "δ": 0.15, "ε": 0.10}

    # Ajustes por abstracción
    if abstraccion == "alta":
        pesos["β"] += 0.05
        pesos["γ"] -= 0.05
    elif abstraccion == "baja":
        pesos["α"] += 0.05

    # Ajustes por factualidad
    if factualidad == "alta":
        pesos["γ"] += 0.10
    elif factualidad == "baja":
        pesos["γ"] -= 0.05
        pesos["α"] += 0.05

    # Ajustes por densidad metafórica
    if densidad == "alta":
        pesos["α"] += 0.05
        pesos["β"] += 0.05
    elif densidad == "baja":
        pesos["α"] -= 0.05

    # Ajustes por intención
    if intencion == "reflexiva":
        pesos["β"] += 0.05
    elif intencion == "disruptiva":
        pesos["α"] += 0.05
        pesos["β"] -= 0.05

    # Ajustes por estilo extra
    if "ligero" in estilo_extra.lower():
        pesos["α"] += 0.05
    if "lírico" in estilo_extra.lower():
        pesos["β"] += 0.05

    # Ajustes por restricciones
    if "sin tecnicismos" in restricciones.lower():
        pesos["γ"] -= 0.05

    # Ajustes por extensión
    if extension == "corta":
        pesos["α"] += 0.05
        pesos["γ"] -= 0.05
    elif extension == "larga":
        pesos["β"] += 0.05

    # Limitar γ a mínimo 0
    pesos["γ"] = max(0, pesos["γ"])

    # Normalizar
    pesos = normalizar(pesos)

    return pesos
