"""Motor de análisis estadístico (CU5, RF-13 a RF-19).

Reglas:
- Normalidad: prueba de Shapiro-Wilk (alfa = 0,05).
- Correlación: Pearson si ambas variables son normales; de lo contrario, Spearman.
- Subgrupos: ANOVA de un factor si todos los grupos son normales y hay homogeneidad de varianzas
  (Levene); de lo contrario, Kruskal-Wallis.
"""
from datetime import datetime

import pandas as pd
from flask import current_app
from scipy import stats

from modelos import (
    Analisis, EstadisticoDescriptivo, ResultadoCorrelacion, ResultadoSubgrupo, bd, rango_experiencia,
    RANGOS_EXPERIENCIA,
)
from servicios.emparejamiento import PrecondicionFaltante, ultimo_emparejamiento

DIMENSIONES = ["apertura", "responsabilidad", "extraversion", "amabilidad", "neuroticismo"]
VARIABLES_RENDIMIENTO = ["rendimiento_autopercibido", "story_points", "issues_completados", "bugs_resueltos",
                         "tiempo_ciclo_dias"]
VARIABLES_NUMERICAS = DIMENSIONES + VARIABLES_RENDIMIENTO


def construir_tabla_emparejada(emparejamiento) -> pd.DataFrame:
    """Base consolidada lista para el análisis (una fila por participante emparejado)."""
    filas = []
    for registro in emparejamiento.registros:
        participante, metrica, puntaje = registro.participante, registro.metrica, registro.participante.puntaje
        filas.append({
            "codigo_anonimo": participante.codigo_anonimo,
            "rol": participante.rol_profesional,
            "experiencia": rango_experiencia(participante.anios_experiencia),
            "metodologia": participante.equipo.metodologia_trabajo,
            **{dimension: getattr(puntaje, dimension) for dimension in DIMENSIONES},
            "rendimiento_autopercibido": puntaje.rendimiento_autopercibido,
            "story_points": metrica.story_points,
            "issues_completados": metrica.issues_completados,
            "bugs_resueltos": metrica.bugs_resueltos,
            "tiempo_ciclo_dias": metrica.tiempo_ciclo_dias,
        })
    return pd.DataFrame(filas)


def es_normal(serie: pd.Series, alfa: float) -> bool:
    serie = serie.dropna()
    if len(serie) < 3 or serie.nunique() < 2:
        return False
    return bool(stats.shapiro(serie).pvalue > alfa)


def correlacionar(x: pd.Series, y: pd.Series, alfa: float) -> tuple[str, float, float] | None:
    """Devuelve (prueba, coeficiente, p) o None si una variable es constante."""
    if x.nunique() < 2 or y.nunique() < 2:
        return None
    if es_normal(x, alfa) and es_normal(y, alfa):
        coeficiente, p = stats.pearsonr(x, y)
        return "Pearson", float(coeficiente), float(p)
    coeficiente, p = stats.spearmanr(x, y)
    return "Spearman", float(coeficiente), float(p)


def comparar_grupos(tabla: pd.DataFrame, agrupacion: str, variable: str, minimo_grupo: int, alfa: float):
    """Devuelve (prueba, estadístico, p, n_grupos, excluidos) o None si no hay ≥2 grupos válidos."""
    grupos, excluidos = [], []
    for categoria, subtabla in tabla.groupby(agrupacion):
        if len(subtabla) >= minimo_grupo:
            grupos.append(subtabla[variable].astype(float))
        else:
            excluidos.append(f"{categoria} (n={len(subtabla)})")
    if len(grupos) < 2 or pd.concat(grupos).nunique() < 2:
        return None
    supuestos = all(es_normal(g, alfa) for g in grupos) and stats.levene(*grupos).pvalue > alfa
    if supuestos:
        estadistico, p = stats.f_oneway(*grupos)
        prueba = "ANOVA"
    else:
        estadistico, p = stats.kruskal(*grupos)
        prueba = "Kruskal-Wallis"
    if pd.isna(estadistico) or pd.isna(p):
        return None
    return prueba, float(estadistico), float(p), len(grupos), excluidos


def calcular_descriptivos(tabla: pd.DataFrame) -> list[EstadisticoDescriptivo]:
    filas = []

    def agregar(agrupacion, categoria, subtabla):
        for variable in VARIABLES_NUMERICAS:
            desviacion = subtabla[variable].std(ddof=1) if len(subtabla) > 1 else None
            filas.append(EstadisticoDescriptivo(
                agrupacion=agrupacion, categoria=categoria, variable=variable, frecuencia=len(subtabla),
                media=float(subtabla[variable].mean()),
                desviacion=None if desviacion is None or pd.isna(desviacion) else float(desviacion)))

    agregar("global", "Muestra total", tabla)
    for agrupacion in ("rol", "experiencia", "metodologia"):
        for categoria, subtabla in tabla.groupby(agrupacion):
            agregar(agrupacion, str(categoria), subtabla)
    return filas


def ejecutar_analisis() -> Analisis:
    emparejamiento = ultimo_emparejamiento()
    if emparejamiento is None:
        raise PrecondicionFaltante("Aún no se ha ejecutado el emparejamiento de datos.",
                                   "Ejecute primero el emparejamiento.", "/admin/emparejamiento")
    if emparejamiento.n_emparejados == 0:
        raise PrecondicionFaltante("El emparejamiento no produjo ningún registro emparejado.",
                                   "Verifique que los códigos del reporte coincidan con los del formulario.",
                                   "/admin/emparejamiento")

    configuracion = current_app.config
    alfa = configuracion["NIVEL_SIGNIFICANCIA"]
    minimo_muestra = configuracion["MUESTRA_MINIMA_EMPAREJADOS"]
    minimo_grupo = configuracion["MUESTRA_MINIMA_SUBGRUPO"]

    tabla = construir_tabla_emparejada(emparejamiento)
    advertencias = []
    analisis = Analisis(id_emparejamiento=emparejamiento.id_emparejamiento, n_muestra=len(tabla), estado="completo")

    analisis.descriptivos.extend(calcular_descriptivos(tabla))

    if len(tabla) < minimo_muestra:
        analisis.estado = "datos_insuficientes"
        advertencias.append(
            f"Muestra insuficiente: {len(tabla)} registros emparejados (mínimo {minimo_muestra}). "
            "Se calculó solo la estadística descriptiva; amplíe la muestra para el análisis inferencial.")
    else:
        for dimension in DIMENSIONES:
            for variable in VARIABLES_RENDIMIENTO:
                resultado = correlacionar(tabla[dimension], tabla[variable], alfa)
                if resultado is None:
                    advertencias.append(f"No se correlacionó {dimension} con {variable}: una de las variables es constante.")
                    continue
                prueba, coeficiente, p = resultado
                analisis.correlaciones.append(ResultadoCorrelacion(
                    dimension=dimension, variable_rendimiento=variable, prueba=prueba,
                    coeficiente=coeficiente, p_valor=p, n=len(tabla)))
        for agrupacion in ("rol", "experiencia"):
            for variable in DIMENSIONES + ["rendimiento_autopercibido"]:
                resultado = comparar_grupos(tabla, agrupacion, variable, minimo_grupo, alfa)
                if resultado is None:
                    advertencias.append(f"Comparación por {agrupacion} omitida para {variable}: menos de 2 subgrupos con "
                                        f"al menos {minimo_grupo} participantes.")
                    continue
                prueba, estadistico, p, n_grupos, excluidos = resultado
                analisis.subgrupos.append(ResultadoSubgrupo(
                    variable_agrupacion=agrupacion, variable_comparada=variable, prueba=prueba,
                    estadistico=estadistico, p_valor=p, grupos_incluidos=n_grupos,
                    grupos_excluidos="; ".join(excluidos)))
        excluidos_unicos = sorted({e for s in analisis.subgrupos for e in s.grupos_excluidos.split("; ") if e})
        if excluidos_unicos:
            advertencias.append("Subgrupos excluidos del análisis inferencial por tener pocos participantes: "
                                + ", ".join(excluidos_unicos) + ".")

    analisis.advertencias = "\n".join(advertencias)
    analisis.fecha_ejecucion = datetime.now()
    bd.session.add(analisis)
    bd.session.commit()
    return analisis


def ultimo_analisis() -> Analisis | None:
    return Analisis.query.order_by(Analisis.id_analisis.desc()).first()
