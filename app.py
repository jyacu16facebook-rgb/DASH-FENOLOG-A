# app.py
# ==========================================================
# DASH: Fenología y estructura vs rendimiento (KG/HA)
# - Ponderaciones por Ha COSECHADA donde corresponde
# - Sumatorias "kilogramos" por campaña (sin ponderar)
# - % cuajado cappeado a 100%
# - Filtro EDAD PLANTA FINAL
# ==========================================================

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Fenología vs rendimiento", layout="wide")

TITLE = "🫐 Fenología y estructura vs rendimiento (KG/HA) | Campañas 2022–2025"
st.title(TITLE)

# -----------------------------
# Helpers
# -----------------------------
def to_numeric(series: pd.Series) -> pd.Series:
    """Convierte a numérico tolerando strings y NaN."""
    return pd.to_numeric(series, errors="coerce")


def safe_div(n, d):
    n = to_numeric(n)
    d = to_numeric(d)
    return np.where((d > 0) & np.isfinite(d), n / d, np.nan)


def wavg(df, value_col, weight_col):
    """Promedio ponderado robusto (ignora NaN y pesos <= 0)."""
    if value_col not in df.columns or weight_col not in df.columns:
        return np.nan
    v = to_numeric(df[value_col])
    w = to_numeric(df[weight_col])
    m = v.notna() & w.notna() & (w > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.average(v[m], weights=w[m]))


def wsum(df, value_col, weight_col):
    """Suma ponderada: sum(value * weight) donde weight > 0 y ambos válidos."""
    v = to_numeric(df[value_col])
    w = to_numeric(df[weight_col])
    m = v.notna() & w.notna() & (w > 0)
    if m.sum() == 0:
        return np.nan
    return float((v[m] * w[m]).sum())


def enforce_campaign_order(x):
    """
    Fuerza campaña como categoría ordenada para evitar ticks tipo 2022.5.
    """
    order = ["2022", "2023", "2024", "2025"]
    x = x.astype(str)
    return pd.Categorical(x, categories=order, ordered=True)


def enforce_edad_final_order(x):
    """
    EDAD PLANTA FINAL debe ser 1,2,3+ (como strings) para ejes limpios.
    """
    order = ["1", "2", "3+"]
    x = x.astype(str)
    x = x.replace({"3.0+": "3+", "3+": "3+", "3": "3"})
    x = x.replace({"3": "3+"})  # normaliza por si llega '3'
    return pd.Categorical(x, categories=order, ordered=True)


def build_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sidebar: filtros principales.
    """
    st.sidebar.header("Filtros")

    # Aseguramos tipos "bonitos"
    if "CAMPAÑA" in df.columns:
        df["CAMPAÑA"] = enforce_campaign_order(df["CAMPAÑA"])
    if "EDAD PLANTA FINAL" in df.columns:
        df["EDAD PLANTA FINAL"] = enforce_edad_final_order(df["EDAD PLANTA FINAL"])

    def multiselect_filter(col, label=None):
        nonlocal df
        if col not in df.columns:
            return
        label = label or col
        opts = [x for x in df[col].dropna().astype(str).unique().tolist()]
        opts_sorted = sorted(opts)
        selected = st.sidebar.multiselect(label, options=opts_sorted, default=opts_sorted)
        if selected:
            df = df[df[col].astype(str).isin(selected)]

    # Orden de filtros
    multiselect_filter("FUNDO")
    multiselect_filter("ETAPA")
    multiselect_filter("CAMPO")
    multiselect_filter("TURNO")
    multiselect_filter("VARIEDAD")
    multiselect_filter("EDAD PLANTA FINAL")  # NUEVO filtro solicitado
    multiselect_filter("SIEMBRA")

    # Semanas
    if "SEMANA" in df.columns:
        s = to_numeric(df["SEMANA"])
        if s.notna().any():
            min_s, max_s = int(s.min()), int(s.max())
            sem_range = st.sidebar.slider("Rango de SEMANA", min_value=min_s, max_value=max_s, value=(min_s, max_s))
            df = df[s.between(sem_range[0], sem_range[1], inclusive="both")]

    # Campaña
    if "CAMPAÑA" in df.columns:
        camp_opts = [c for c in df["CAMPAÑA"].cat.categories if c in df["CAMPAÑA"].astype(str).unique()]
        camp_selected = st.sidebar.multiselect("CAMPAÑA", options=camp_opts, default=camp_opts)
        if camp_selected:
            df = df[df["CAMPAÑA"].astype(str).isin([str(x) for x in camp_selected])]

    return df


@st.cache_data(show_spinner=False)
def load_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name="DATA", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    # Tipos base
    numeric_cols = [
        "SEMANA",
        "kilogramos",
        "FLORES",
        "FRUTO CUAJADO",
        "FRUTO VERDE",
        "TOTAL DE FRUTOS",
        "Ha COSECHADA",
        "Ha TURNO",
        "KG/HA",
        "DENSIDAD",
        "FRUTO MADURO",
        "FRUTO ROSADO",
        "FRUTO CREMOSO",
        "PESO BAYA (g)",
        "PESO BAYA CREMOSO (g)",
        "CALIBRE BAYA (mm)",
        "CALIBRE CREMOSO (mm)",
        "SEMANA DE SIEMBRA",
        "MADERAS PRINCIPALES",
        "CORTES",
        "BROTES TOTALES",
        "TERMINALES",
        "EDAD PLANTA",
        "BP_N_BROTES_ULT",
        "BP_LONG_B1_ULT",
        "BP_LONG_B2_ULT",
        "BP_DIAM_B1_ULT",
        "BP_DIAM_B2_ULT",
        "BS_N_BROTES_ULT",
        "BS_LONG_B1_ULT",
        "BS_LONG_B2_ULT",
        "BS_DIAM_B1_ULT",
        "BS_DIAM_B2_ULT",
        "BT_N_BROTES_ULT",
        "BT_LONG_B1_ULT",
        "BT_LONG_B2_ULT",
        "BT_DIAM_B1_ULT",
        "BT_DIAM_B2_ULT",
        "ALTURA_PLANTA_ULT",
        "ANCHO_PLANTA_ULT",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = to_numeric(df[c])

    # Normalizaciones de categorías
    if "CAMPAÑA" in df.columns:
        df["CAMPAÑA"] = enforce_campaign_order(df["CAMPAÑA"].astype(str))

    if "EDAD PLANTA FINAL" in df.columns:
        # Normalizar (si llega 3 en vez de 3+)
        df["EDAD PLANTA FINAL"] = df["EDAD PLANTA FINAL"].astype(str).replace({"3": "3+"})
        df["EDAD PLANTA FINAL"] = enforce_edad_final_order(df["EDAD PLANTA FINAL"])

    return df


def campaign_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resumen por campaña:
    - KG: SUM(kilogramos)  (sin ponderar)
    - KG/HA: promedio ponderado por Ha COSECHADA
    - PESO y CALIBRE: promedio ponderado por Ha COSECHADA
    - AREA_EJECUTADA: SUM(Ha COSECHADA)
    """
    if df.empty:
        return pd.DataFrame()

    out = []
    for camp, g in df.groupby(df["CAMPAÑA"].astype(str), dropna=False):
        row = {
            "CAMPAÑA": camp,
            "KG": float(to_numeric(g["kilogramos"]).fillna(0).sum()) if "kilogramos" in g.columns else np.nan,
            "KG/HA": wavg(g, "KG/HA", "Ha COSECHADA"),
            "PESO (g)": wavg(g, "PESO BAYA (g)", "Ha COSECHADA"),
            "CALIBRE (mm)": wavg(g, "CALIBRE BAYA (mm)", "Ha COSECHADA"),
            "AREA_EJECUTADA (Ha COSECHADA)": float(to_numeric(g["Ha COSECHADA"]).fillna(0).sum()) if "Ha COSECHADA" in g.columns else np.nan,
        }
        out.append(row)

    res = pd.DataFrame(out)
    # Ordena campañas
    res["CAMPAÑA"] = enforce_campaign_order(res["CAMPAÑA"].astype(str))
    res = res.sort_values("CAMPAÑA")
    res["CAMPAÑA"] = res["CAMPAÑA"].astype(str)
    return res.reset_index(drop=True)


def plot_campaign_bar(df_sum: pd.DataFrame, y_col: str, title: str):
    if df_sum.empty or y_col not in df_sum.columns:
        st.info("No hay data suficiente para graficar.")
        return
    fig = px.bar(
        df_sum,
        x="CAMPAÑA",
        y=y_col,
        category_orders={"CAMPAÑA": ["2022", "2023", "2024", "2025"]},
        title=title,
    )
    fig.update_layout(xaxis_title="CAMPAÑA", yaxis_title=y_col)
    st.plotly_chart(fig, use_container_width=True)


def weekly_curve(df: pd.DataFrame):
    if df.empty or "SEMANA" not in df.columns:
        st.info("No hay data suficiente para curva semanal.")
        return

    g = (
        df.groupby(["CAMPAÑA", "SEMANA"], dropna=False)
        .apply(lambda x: pd.Series({"KG_HA_POND": wavg(x, "KG/HA", "Ha COSECHADA")}))
        .reset_index()
    )
    g["CAMPAÑA"] = g["CAMPAÑA"].astype(str)

    fig = px.line(
        g.sort_values(["CAMPAÑA", "SEMANA"]),
        x="SEMANA",
        y="KG_HA_POND",
        color="CAMPAÑA",
        markers=True,
        title="Curva semanal de KG/HA (ponderado por Ha COSECHADA) | Comparación por campaña",
    )
    fig.update_layout(xaxis_title="SEMANA", yaxis_title="KG/HA (pond)")
    st.plotly_chart(fig, use_container_width=True)


def boxplots_edad_y_siembra(df: pd.DataFrame):
    c1, c2 = st.columns(2)

    with c1:
        if "SIEMBRA" in df.columns and df["SIEMBRA"].notna().any():
            gg = (
                df.groupby("SIEMBRA", dropna=False)
                .apply(lambda x: pd.Series({"KG_HA_POND": wavg(x, "KG/HA", "Ha COSECHADA")}))
                .reset_index()
            )
            fig = px.bar(
                gg.sort_values("KG_HA_POND", ascending=False),
                x="SIEMBRA",
                y="KG_HA_POND",
                title="KG/HA ponderado por SIEMBRA",
            )
            fig.update_layout(xaxis_title="SIEMBRA", yaxis_title="KG/HA (pond)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay SIEMBRA disponible para graficar.")

    with c2:
        if "EDAD PLANTA FINAL" in df.columns and df["EDAD PLANTA FINAL"].notna().any():
            # Boxplot (como pediste)
            tmp = df.copy()
            tmp["EDAD PLANTA FINAL"] = tmp["EDAD PLANTA FINAL"].astype(str).replace({"3": "3+"})
            tmp["EDAD PLANTA FINAL"] = pd.Categorical(tmp["EDAD PLANTA FINAL"], categories=["1", "2", "3+"], ordered=True)

            # KG/HA ponderado por TURNO dentro de cada edad (mejor para boxplot)
            agg = (
                tmp.groupby(["EDAD PLANTA FINAL", "TURNO"], dropna=False)
                .apply(lambda x: pd.Series({"KG_HA_POND": wavg(x, "KG/HA", "Ha COSECHADA")}))
                .reset_index()
            )

            fig = px.box(
                agg,
                x="EDAD PLANTA FINAL",
                y="KG_HA_POND",
                category_orders={"EDAD PLANTA FINAL": ["1", "2", "3+"]},
                title="Distribución de KG/HA ponderado por EDAD PLANTA FINAL (boxplot)",
                points="outliers",
            )
            fig.update_layout(xaxis_title="EDAD PLANTA FINAL", yaxis_title="KG/HA (pond)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay EDAD PLANTA FINAL disponible para graficar.")


def variedades_section(df: pd.DataFrame):
    st.subheader("Variedades: ranking (KG/HA ponderado) + VS por campañas")

    if df.empty or "VARIEDAD" not in df.columns:
        st.info("No hay data de VARIEDAD.")
        return

    top_n = st.slider("Top N variedades por frecuencia", min_value=5, max_value=25, value=10)

    # Agregado por variedad: KG/HA ponderado + frecuencia (n) + suma de Ha
    agg = (
        df.groupby("VARIEDAD", dropna=False)
        .apply(
            lambda x: pd.Series(
                {
                    "KG_HA_POND": wavg(x, "KG/HA", "Ha COSECHADA"),
                    "Ha COSECHADA (sum)": float(to_numeric(x["Ha COSECHADA"]).fillna(0).sum()) if "Ha COSECHADA" in x.columns else np.nan,
                    "n": int(len(x)),
                }
            )
        )
        .reset_index()
    )

    top_vars = agg.sort_values("n", ascending=False).head(top_n)["VARIEDAD"].astype(str).tolist()

    c1, c2 = st.columns([1, 2])

    with c1:
        # tabla pequeña solo de apoyo (no la grande que pediste eliminar)
        st.dataframe(
            agg[agg["VARIEDAD"].astype(str).isin(top_vars)].sort_values("n", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        fig = px.bar(
            agg[agg["VARIEDAD"].astype(str).isin(top_vars)].sort_values("KG_HA_POND", ascending=True),
            x="KG_HA_POND",
            y="VARIEDAD",
            orientation="h",
            title="Promedio KG/HA ponderado (Top variedades por frecuencia)",
        )
        fig.update_layout(xaxis_title="KG/HA (pond)", yaxis_title="VARIEDAD")
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap VARIEDAD x CAMPAÑA
    if "CAMPAÑA" in df.columns:
        df2 = df[df["VARIEDAD"].astype(str).isin(top_vars)].copy()
        pivot = (
            df2.groupby(["VARIEDAD", "CAMPAÑA"], dropna=False)
            .apply(lambda x: pd.Series({"KG_HA_POND": wavg(x, "KG/HA", "Ha COSECHADA")}))
            .reset_index()
        )
        pivot["CAMPAÑA"] = pivot["CAMPAÑA"].astype(str)

        heat = pivot.pivot(index="VARIEDAD", columns="CAMPAÑA", values="KG_HA_POND")
        heat = heat.reindex(columns=["2022", "2023", "2024", "2025"])

        fig2 = px.imshow(
            heat,
            aspect="auto",
            title="VS: VARIEDAD x CAMPAÑA (KG/HA ponderado)",
        )
        st.plotly_chart(fig2, use_container_width=True)


def best_worst_turno_with_structure(df: pd.DataFrame):
    st.subheader("Best vs Worst TURNO dentro de (CAMPAÑA + VARIEDAD)")

    if df.empty or any(c not in df.columns for c in ["CAMPAÑA", "VARIEDAD", "TURNO"]):
        st.info("Faltan columnas para Best/Worst.")
        return

    # Selector de variedad (sobre lo ya filtrado)
    variedades = sorted(df["VARIEDAD"].dropna().astype(str).unique().tolist())
    if not variedades:
        st.info("No hay variedades disponibles.")
        return
    var_sel = st.selectbox("VARIEDAD", variedades, index=0)

    sub = df[df["VARIEDAD"].astype(str) == var_sel].copy()
    if sub.empty:
        st.info("No hay data para la variedad seleccionada.")
        return

    # Agregamos a nivel TURNO por CAMPAÑA (y mantenemos ETAPA/CAMPO como etiqueta de contexto)
    # Para cada TURNO: KG/HA pond + estructura pond + "ETAPA/CAMPO más frecuente"
    def mode_str(s):
        s = s.dropna().astype(str)
        if s.empty:
            return ""
        return s.value_counts().index[0]

    per_turno = (
        sub.groupby(["CAMPAÑA", "TURNO"], dropna=False)
        .apply(
            lambda x: pd.Series(
                {
                    "KG_HA_POND": wavg(x, "KG/HA", "Ha COSECHADA"),
                    "MADERAS_POND": wavg(x, "MADERAS PRINCIPALES", "Ha COSECHADA"),
                    "CORTES_POND": wavg(x, "CORTES", "Ha COSECHADA"),
                    "BROTES_POND": wavg(x, "BROTES TOTALES", "Ha COSECHADA"),
                    "ETAPA_ctx": mode_str(x["ETAPA"]) if "ETAPA" in x.columns else "",
                    "CAMPO_ctx": mode_str(x["CAMPO"]) if "CAMPO" in x.columns else "",
                }
            )
        )
        .reset_index()
    )
    per_turno["CAMPAÑA"] = per_turno["CAMPAÑA"].astype(str)

    # Para cada campaña: encontrar TURNO max y min (por KG_HA_POND)
    rows = []
    for camp, g in per_turno.groupby("CAMPAÑA", dropna=False):
        g2 = g.dropna(subset=["KG_HA_POND"])
        if g2.empty:
            continue

        max_row = g2.loc[g2["KG_HA_POND"].idxmax()]
        min_row = g2.loc[g2["KG_HA_POND"].idxmin()]

        rows.append(
            {
                "CAMPAÑA": camp,
                "TURNO_MAX": max_row["TURNO"],
                "ETAPA_MAX": max_row.get("ETAPA_ctx", ""),
                "CAMPO_MAX": max_row.get("CAMPO_ctx", ""),
                "KG/HA_MAX": max_row["KG_HA_POND"],
                "MADERAS_MAX": max_row["MADERAS_POND"],
                "CORTES_MAX": max_row["CORTES_POND"],
                "BROTES_TOTALES_MAX": max_row["BROTES_POND"],
                "TURNO_MIN": min_row["TURNO"],
                "ETAPA_MIN": min_row.get("ETAPA_ctx", ""),
                "CAMPO_MIN": min_row.get("CAMPO_ctx", ""),
                "KG/HA_MIN": min_row["KG_HA_POND"],
                "MADERAS_MIN": min_row["MADERAS_POND"],
                "CORTES_MIN": min_row["CORTES_POND"],
                "BROTES_TOTALES_MIN": min_row["BROTES_POND"],
            }
        )

    bw = pd.DataFrame(rows)
    if bw.empty:
        st.info("No se pudo calcular Best/Worst con la data actual.")
        return

    bw["CAMPAÑA"] = pd.Categorical(bw["CAMPAÑA"].astype(str), categories=["2022", "2023", "2024", "2025"], ordered=True)
    bw = bw.sort_values("CAMPAÑA").reset_index(drop=True)
    bw_show = bw.copy()
    bw_show["CAMPAÑA"] = bw_show["CAMPAÑA"].astype(str)

    st.dataframe(bw_show, use_container_width=True, hide_index=True)

    # Gráfico estilo “imagen”: barras KG/HA + líneas estructura (eje secundario)
    camp_sel = st.selectbox("Ver detalle de CAMPAÑA", bw_show["CAMPAÑA"].unique().tolist(), index=0)
    row = bw_show[bw_show["CAMPAÑA"] == camp_sel].iloc[0]

    labels = [
        f"{var_sel}<br>{row['TURNO_MAX']}<br>{row['CAMPO_MAX']}<br>{row['ETAPA_MAX']}",
        f"{var_sel}<br>{row['TURNO_MIN']}<br>{row['CAMPO_MIN']}<br>{row['ETAPA_MIN']}",
    ]
    kg_vals = [row["KG/HA_MAX"], row["KG/HA_MIN"]]
    maderas = [row["MADERAS_MAX"], row["MADERAS_MIN"]]
    cortes = [row["CORTES_MAX"], row["CORTES_MIN"]]
    brotes = [row["BROTES_TOTALES_MAX"], row["BROTES_TOTALES_MIN"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="KG/HA (pond)", x=labels, y=kg_vals))
    fig.add_trace(go.Scatter(name="MADERAS PRINCIPALES (pond)", x=labels, y=maderas, yaxis="y2", mode="lines+markers"))
    fig.add_trace(go.Scatter(name="CORTES (pond)", x=labels, y=cortes, yaxis="y2", mode="lines+markers"))
    fig.add_trace(go.Scatter(name="BROTES TOTALES (pond)", x=labels, y=brotes, yaxis="y2", mode="lines+markers"))

    fig.update_layout(
        title=f"KG/HA MAX vs MIN (por TURNO) + Estructura | CAMPAÑA {camp_sel}",
        yaxis=dict(title="KG/HA (pond)"),
        yaxis2=dict(title="Estructura (pond)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


def correlation_heatmap(df: pd.DataFrame):
    st.subheader("Mapa de correlaciones (solo variables seleccionadas)")

    cols = [
        "KG/HA",
        "kilogramos",
        "FLORES",
        "Ha COSECHADA",
        "DENSIDAD",
        "FRUTO MADURO",
        "PESO BAYA (g)",
        "CALIBRE BAYA (mm)",
        "SEMANA DE SIEMBRA",
        "MADERAS PRINCIPALES",
        "CORTES",
        "BROTES TOTALES",
        "TERMINALES",
        "EDAD PLANTA",
    ]
    cols = [c for c in cols if c in df.columns]

    if len(cols) < 3:
        st.info("No hay suficientes columnas numéricas para correlaciones.")
        return

    dd = df[cols].copy()
    for c in cols:
        dd[c] = to_numeric(dd[c])

    # Requiere data suficiente
    dd = dd.dropna(how="all")
    if dd.shape[0] < 10:
        st.info("Muy pocos datos para correlaciones.")
        return

    corr = dd.corr(numeric_only=True)

    fig = px.imshow(corr, aspect="auto", title="Correlación (Pearson) - variables seleccionadas")
    st.plotly_chart(fig, use_container_width=True)


def cuajado_section(df: pd.DataFrame):
    st.subheader("% Cuajado por campaña (cappeado a 100%)")

    if df.empty or any(c not in df.columns for c in ["CAMPAÑA", "FLORES", "FRUTO CUAJADO"]):
        st.info("Faltan columnas para % cuajado.")
        return

    g = (
        df.groupby("CAMPAÑA", dropna=False)
        .agg(
            FLORES_SUM=("FLORES", "sum"),
            CUAJADO_SUM=("FRUTO CUAJADO", "sum"),
        )
        .reset_index()
    )
    g["CAMPAÑA"] = g["CAMPAÑA"].astype(str)

    g["%_CUAJADO"] = (g["CUAJADO_SUM"] / g["FLORES_SUM"]) * 100
    g["%_CUAJADO"] = g["%_CUAJADO"].replace([np.inf, -np.inf], np.nan)
    g["%_CUAJADO"] = g["%_CUAJADO"].clip(lower=0, upper=100)  # <<< cap a 100

    fig = px.bar(
        g,
        x="CAMPAÑA",
        y="%_CUAJADO",
        category_orders={"CAMPAÑA": ["2022", "2023", "2024", "2025"]},
        title="% Cuajado por campaña (sum Cuajado / sum Flores) | máx 100%",
        hover_data=["FLORES_SUM", "CUAJADO_SUM"],
    )
    fig.update_layout(xaxis_title="CAMPAÑA", yaxis_title="%_CUAJADO")
    st.plotly_chart(fig, use_container_width=True)


def model_importance_section(df: pd.DataFrame):
    st.subheader("Variables más asociadas a KG/HA (modelo)")

    if df.empty or "KG/HA" not in df.columns:
        st.info("No hay data suficiente para el modelo.")
        return

    top_k = st.slider("Top K variables (importancia)", min_value=10, max_value=40, value=20)

    # Features candidatas (mezcla numéricas + categóricas útiles)
    candidate_features = [
        # categóricas
        "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD", "SIEMBRA",
        "SEMANA", "SEMANA DE SIEMBRA", "CAMPAÑA",
        # numéricas clave
        "kilogramos", "Ha COSECHADA", "Ha TURNO", "DENSIDAD",
        "FLORES", "FRUTO CUAJADO", "FRUTO VERDE", "TOTAL DE FRUTOS",
        "FRUTO MADURO", "FRUTO ROSADO", "FRUTO CREMOSO",
        "PESO BAYA (g)", "CALIBRE BAYA (mm)",
        "MADERAS PRINCIPALES", "CORTES", "BROTES TOTALES", "TERMINALES",
        "EDAD PLANTA", "EDAD PLANTA FINAL",
        "ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT",
    ]
    candidate_features = [c for c in candidate_features if c in df.columns]
    if len(candidate_features) < 3:
        st.info("Muy pocas variables disponibles para entrenar.")
        return

    d = df[candidate_features + ["KG/HA"]].copy()

    # Target
    y = to_numeric(d["KG/HA"])
    d = d.drop(columns=["KG/HA"])
    # Elimina filas con target NaN
    m = y.notna()
    X = d.loc[m].copy()
    y = y.loc[m].copy()

    # Si queda poca data, salimos
    if len(X) < 200:
        st.info("Poca data para modelo (necesito al menos ~200 filas útiles).")
        return

    # Identificar tipos
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Preprocess
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=3,
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    with st.spinner("Entrenando modelo e importancias (permutation importance)..."):
        pipe.fit(X_train, y_train)

        # Permutation importance sobre test
        r = permutation_importance(
            pipe,
            X_test,
            y_test,
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
            scoring="r2",
        )

    # Nombres de features transformadas
    # - num: mismos nombres
    # - cat: onehot names
    feature_names = []
    feature_names.extend(num_cols)
    if cat_cols:
        ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(ohe_names)

    importances = pd.DataFrame({
        "feature": feature_names,
        "imp": r.importances_mean
    }).sort_values("imp", ascending=False)

    # Agregar por variable original (para que sea legible)
    def base_feature(name: str) -> str:
        # ejemplo: VARIEDAD_SEKOYA POP -> VARIEDAD
        return name.split("_")[0]

    importances["base_feature"] = importances["feature"].map(base_feature)
    agg = importances.groupby("base_feature", as_index=False)["imp"].sum().sort_values("imp", ascending=False).head(top_k)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(agg.reset_index(drop=True), use_container_width=True, hide_index=True)
    with c2:
        fig = px.bar(
            agg.sort_values("imp", ascending=True),
            x="imp",
            y="base_feature",
            orientation="h",
            title="Importancia global (agregada por variable original)",
        )
        fig.update_layout(xaxis_title="Importancia (suma)", yaxis_title="Variable")
        st.plotly_chart(fig, use_container_width=True)


def scatter_section_fixed_turno(df: pd.DataFrame):
    st.subheader("Dispersión (X vs Métrica ponderada) + BEST/WORST (Nivel fijo: TURNO)")

    if df.empty:
        st.info("No hay data para dispersión.")
        return

    # Métrica Y (ponderada)
    metric = st.selectbox(
        "Métrica (ponderada por Ha COSECHADA)",
        options=["KG/HA", "PESO BAYA (g)", "CALIBRE BAYA (mm)"],
        index=0,
    )

    # Variable X (resto de variables)
    avoid = set(["AÑO", "CAMPAÑA"])  # mantenemos campaña para agrupaciones, pero como X no es útil aquí
    candidates = [c for c in df.columns if c not in avoid and c != metric]
    # Solo variables numéricas para scatter (más limpio)
    numeric_candidates = []
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_candidates.append(c)

    if not numeric_candidates:
        st.info("No encuentro variables numéricas para usar como X.")
        return

    x_var = st.selectbox("Variable X", options=sorted(numeric_candidates), index=0)

    # Agregación a nivel TURNO (y campaña para marcar best/worst dentro del conjunto filtrado completo)
    gb = (
        df.groupby(["CAMPAÑA", "TURNO"], dropna=False)
        .apply(
            lambda x: pd.Series(
                {
                    "Y_POND": wavg(x, metric, "Ha COSECHADA"),
                    "X_POND": wavg(x, x_var, "Ha COSECHADA"),
                    "Ha_SUM": float(to_numeric(x["Ha COSECHADA"]).fillna(0).sum()) if "Ha COSECHADA" in x.columns else np.nan,
                }
            )
        )
        .reset_index()
    )
    gb["CAMPAÑA"] = gb["CAMPAÑA"].astype(str)

    # BEST/WORST global (en lo filtrado)
    gb2 = gb.dropna(subset=["Y_POND", "X_POND"])
    if gb2.empty:
        st.info("No hay data suficiente para dispersión.")
        return

    best_idx = gb2["Y_POND"].idxmax()
    worst_idx = gb2["Y_POND"].idxmin()

    gb2["POINT"] = "NORMAL"
    gb2.loc[best_idx, "POINT"] = "BEST"
    gb2.loc[worst_idx, "POINT"] = "WORST"

    fig = px.scatter(
        gb2,
        x="X_POND",
        y="Y_POND",
        color="POINT",
        hover_data=["CAMPAÑA", "TURNO", "Ha_SUM"],
        title=f"{x_var} vs {metric} (ponderado por Ha COSECHADA) | Nivel: TURNO",
    )
    fig.update_layout(xaxis_title=x_var, yaxis_title=f"{metric} (pond)")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# UI: carga
# -----------------------------
st.sidebar.header("Cargar Excel")
uploaded = st.sidebar.file_uploader("Sube el Excel consolidado (.xlsx)", type=["xlsx"])
st.sidebar.caption("Requisito: la hoja debe llamarse exactamente **DATA**.")

if not uploaded:
    st.warning("📌 Sube tu archivo Excel para empezar.")
    st.stop()

df = load_excel(uploaded)

# -----------------------------
# Filtros
# -----------------------------
df_f = build_filters(df)

# -----------------------------
# Resumen principal
# -----------------------------
st.subheader("Resumen por campaña (ponderado por Ha COSECHADA en promedios)")
df_sum = campaign_summary(df_f)

if df_sum.empty:
    st.error("No hay registros después de aplicar filtros.")
    st.stop()

st.dataframe(df_sum, use_container_width=True, hide_index=True)

# (IMPORTANTE) Eliminamos la “segunda imagen” (no mostramos bar duplicado aquí)
# Si quieres un único gráfico, deja SOLO uno:
plot_campaign_bar(df_sum, "KG/HA", "KG/HA ponderado por Ha COSECHADA (por campaña)")

# -----------------------------
# Dispersión (reemplazo sin “nivel de agregación”)
# -----------------------------
scatter_section_fixed_turno(df_f)

# -----------------------------
# Best/Worst por TURNO dentro de (CAMPAÑA+VARIEDAD) + estructura
# -----------------------------
best_worst_turno_with_structure(df_f)

# -----------------------------
# Curva semanal
# -----------------------------
weekly_curve(df_f)

# -----------------------------
# SIEMBRA + EDAD (boxplot)
# -----------------------------
st.subheader("EDAD PLANTA FINAL y SIEMBRA")
boxplots_edad_y_siembra(df_f)

# -----------------------------
# KG/PLANTA VS campañas (si existe)
# -----------------------------
if "KG/PLANTA" in df_f.columns:
    st.subheader("KG/PLANTA (ponderado por Ha COSECHADA) vs campañas")
    g = (
        df_f.groupby("CAMPAÑA", dropna=False)
        .apply(lambda x: pd.Series({"KG_PLANTA_POND": wavg(x, "KG/PLANTA", "Ha COSECHADA")}))
        .reset_index()
    )
    g["CAMPAÑA"] = g["CAMPAÑA"].astype(str)
    fig = px.line(
        g.sort_values("CAMPAÑA"),
        x="CAMPAÑA",
        y="KG_PLANTA_POND",
        markers=True,
        category_orders={"CAMPAÑA": ["2022", "2023", "2024", "2025"]},
        title="KG/PLANTA ponderado (Ha COSECHADA) vs campañas",
    )
    fig.update_layout(xaxis_title="CAMPAÑA", yaxis_title="KG/PLANTA (pond)")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Variedades: ranking + vs campañas
# -----------------------------
variedades_section(df_f)

# -----------------------------
# % Cuajado (cappeado)
# -----------------------------
cuajado_section(df_f)

# -----------------------------
# Correlaciones (solo columnas solicitadas)
# -----------------------------
correlation_heatmap(df_f)

# -----------------------------
# Modelo: variables más asociadas (IMPORTANCIA)
# -----------------------------
model_importance_section(df_f)
