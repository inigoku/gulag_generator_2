def normalizar(pesos):
    total = sum(pesos.values())
    if total == 0:
        return pesos
    return {k: v / total for k, v in pesos.items()}


# Semàntica dels pesos:
#   α — ancoratge estilístic (obra pròpia, densitat metafòrica)
#   β — context literari (influències, abstracció, reflexió)
#   γ — factualitat externa (Brave Search, referencialitat)
#   δ — folklore / tradició oral
#   ε — context extès (reserva)

PESOS_BASE = {
    "intim":        {"α": 0.45, "β": 0.25, "γ": 0.05, "δ": 0.15, "ε": 0.10},
    "conceptual":   {"α": 0.40, "β": 0.30, "γ": 0.05, "δ": 0.15, "ε": 0.10},
    "experimental": {"α": 0.35, "β": 0.25, "γ": 0.05, "δ": 0.20, "ε": 0.15},
    "_default":     {"α": 0.40, "β": 0.25, "γ": 0.10, "δ": 0.15, "ε": 0.10},
}

AJUSTOS = [
    # (condició, pes, delta)
    # Abstracció
    (lambda p: p.get("nivel_abstraccion") == "alta",  "β", +0.05),
    (lambda p: p.get("nivel_abstraccion") == "alta",  "γ", -0.05),
    (lambda p: p.get("nivel_abstraccion") == "baja",  "α", +0.05),
    # Factualitat
    (lambda p: p.get("grado_factualidad") == "alta",  "γ", +0.10),
    (lambda p: p.get("grado_factualidad") == "baja",  "γ", -0.05),
    (lambda p: p.get("grado_factualidad") == "baja",  "α", +0.05),
    # Densitat metafòrica
    (lambda p: p.get("densidad_metaforica") == "alta", "α", +0.05),
    (lambda p: p.get("densidad_metaforica") == "alta", "β", +0.05),
    (lambda p: p.get("densidad_metaforica") == "baja", "α", -0.05),
    # Intenció
    (lambda p: p.get("intencion_poetica") == "reflexiva",  "β", +0.05),
    (lambda p: p.get("intencion_poetica") == "disruptiva", "α", +0.05),
    (lambda p: p.get("intencion_poetica") == "disruptiva", "β", -0.05),
    # Estil extra
    (lambda p: "ligero"  in p.get("estilo_extra", "").lower(), "α", +0.05),
    (lambda p: "lírico"  in p.get("estilo_extra", "").lower(), "β", +0.05),
    # Restriccions
    (lambda p: "sin tecnicismos" in p.get("restricciones", "").lower(), "γ", -0.05),
    # Extensió
    (lambda p: p.get("extension") == "corta",  "α", +0.05),
    (lambda p: p.get("extension") == "corta",  "γ", -0.05),
    (lambda p: p.get("extension") == "larga",  "β", +0.05),
]


def calcular_pesos(perfil):
    """
    Calcula els pesos adaptatius del agent poètic.

    Retorna un dict amb les claus α β γ δ ε (normalitzades a suma=1)
    i les claus derivades k_obra, k_influencies, k_folklore, k_extendido
    que es poden usar directament com a paràmetres de Chroma.
    """
    categoria = perfil.get("categoria", "_default")
    pesos = dict(PESOS_BASE.get(categoria, PESOS_BASE["_default"]))

    # Aplicar ajustos declaratius
    for condicio, clau, delta in AJUSTOS:
        if condicio(perfil):
            pesos[clau] = pesos.get(clau, 0) + delta

    # Guardar γ_raw per auditoria abans de forçar mínim
    pesos["_γ_raw"] = pesos["γ"]
    pesos["γ"] = max(0.0, pesos["γ"])

    # Normalitzar
    pesos_norm = {k: v for k, v in pesos.items() if not k.startswith("_")}
    pesos_norm = normalizar(pesos_norm)

    # --- NOU: derivar paràmetres de Chroma directament ---
    # Rang: α,β ∈ [0.3, 0.6] → k ∈ [2, 10]
    def _k(pes, k_max):
        return max(1, round(pes * k_max * 2.5))

    pesos_norm["k_obra"]        = _k(pesos_norm["α"], 10)  # màx 10 fragments
    pesos_norm["k_influencies"] = _k(pesos_norm["β"],  8)
    pesos_norm["k_folklore"]    = _k(pesos_norm["δ"],  6)
    pesos_norm["k_extendido"]   = _k(pesos_norm["ε"],  4)

    # Rigidesa estructural (solia llegir-se d'una clau inexistent)
    pesos_norm["rigidez_estructural"] = round(
        0.3 + pesos_norm["α"] * 0.4 + (1 - pesos_norm["β"]) * 0.3, 3
    )

    return pesos_norm