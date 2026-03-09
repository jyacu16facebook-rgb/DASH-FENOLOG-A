# app.py
# ==========================================================
# DASH: Fenología y estructura vs rendimiento (KG/HA)
# - Carga AUTOMÁTICA desde archivo local en repo (xlsx) | Hoja: DATA
# - Filtros: Fundo, Etapa, Campo, Turno, Variedad, Semana, Campaña, EDAD PLANTA FINAL
# - Métricas:
#   * KG = SUMA(kilogramos)  (sin ponderar)
#   * KG/HA, PESO, CALIBRE = promedios ponderados por Ha COSECHADA
#   * Área ejecutada = SUMA(Ha COSECHADA)
# - Scatter: Y = (KG/HA / PESO / CALIBRE) ponderado, X = otras variables
# - Boxplots: por SIEMBRA y por EDAD PLANTA FINAL (1,2,3+)
# - Curva semanal: KG/HA ponderado por Ha COSECHADA (EJE X continuo por campaña)
# - Variedades: ranking ponderado + heatmap VS por campaña
# - Best/Worst TURNO dentro de (CAMPAÑA + VARIEDAD) con ETAPA/CAMPO + estructura (promedios ponderados)
# - Correlaciones: solo columnas solicitadas
# - Vista adicional: KG/PLANTA con fórmula:
#       KG/PLANTA = kilogramos / (Ha TURNO * DENSIDAD)
#   respetando el TURNO como unidad única para evitar duplicar área por semana
# - Vistas agregadas:
#   (1) KG/HA por edad y campaña
#   (2) Densidad vs rendimiento
#   (3) Índice de vigor vegetativo post-poda
# ==========================================================

import os
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

# --------------------------
# CONFIG
# --------------------------
st.set_page_config(page_title="Fenología vs rendimiento", layout="wide")

REQ_SHEET = "DATA"
DATA_FILE = "CONSOLIDADO 2022-2026.xlsx"

COLS_REQUIRED = [
    "AÑO", "CAMPAÑA", "SEMANA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
    "kilogramos", "FLORES", "FRUTO CUAJADO", "FRUTO VERDE", "TOTAL DE FRUTOS",
    "Ha COSECHADA", "Ha TURNO", "KG/HA", "DENSIDAD", "FRUTO MADURO",
    "FRUTO ROSADO", "FRUTO CREMOSO",
    "PESO BAYA (g)", "PESO BAYA CREMOSO (g)",
    "CALIBRE BAYA (mm)", "CALIBRE CREMOSO (mm)",
    "SEMANA DE SIEMBRA", "FECHA FIN DE SIEMBRA", "TIPO PODA", "FECHA PODA",
    "MADERAS PRINCIPALES", "CORTES", "BROTES TOTALES", "TERMINALES",
    "EDAD PLANTA", "EDAD PLANTA FINAL",
    "BP_N_BROTES_ULT", "BP_LONG_ULT", "BP_DIAM_ULT",
    "BS_N_BROTES_ULT", "BS_LONG_ULT", "BS_DIAM_ULT",
    "BT_N_BROTES_ULT", "BT_LONG_ULT", "BT_DIAM_ULT",
    "ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT",
    "SIEMBRA"
]

W_COL = "Ha COSECHADA"

METRIC_Y_OPTIONS = {
    "KG/HA": "KG/HA",
    "PESO BAYA (g)": "PESO BAYA (g)",
    "CALIBRE BAYA (mm)": "CALIBRE BAYA (mm)",
}

STRUCT_COLS = {
    "MADERAS PRINCIPALES": "MADERAS PRINCIPALES",
    "CORTES": "CORTES",
    "BROTES TOTALES": "BROTES TOTALES",
}

CORR_COLS = [
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

# Variables para índice de vigor
VIGOR_VARS = [
    "BP_N_BROTES_ULT",
    "BS_N_BROTES_ULT",
    "BT_N_BROTES_ULT",
    "BP_LONG_ULT",
    "BS_LONG_ULT",
    "BT_LONG_ULT",
    "BP_DIAM_ULT",
    "BS_DIAM_ULT",
    "BT_DIAM_ULT",
    "ALTURA_PLANTA_ULT",
    "ANCHO_PLANTA_ULT",
]

UNIT_COLS_VIGOR = ["CAMPAÑA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]

# --------------------------
# HELPERS
# --------------------------
def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    mask = x.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))

def ensure_categories_age(df: pd.DataFrame) -> pd.DataFrame:
    if "EDAD PLANTA FINAL" in df.columns:
        df["EDAD PLANTA FINAL"] = df["EDAD PLANTA FINAL"].astype(str).str.strip()
        df.loc[df["EDAD PLANTA FINAL"].isin(["3", "3.0", "3.00"]), "EDAD PLANTA FINAL"] = "3+"
        order = ["1", "2", "3+"]
        df["EDAD PLANTA FINAL"] = pd.Categorical(df["EDAD PLANTA FINAL"], categories=order, ordered=True)
    return df

@st.cache_data(show_spinner=False)
def read_excel_path(path: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)

def validate_cols(df: pd.DataFrame) -> list:
    return [c for c in COLS_REQUIRED if c not in df.columns]

def apply_filters(df: pd.DataFrame,
                  camp, fundo, etapa, campo, turno, variedad, edad_final,
                  semana_min, semana_max):
    dff = df.copy()

    if camp:
        dff = dff[dff["CAMPAÑA"].isin(camp)]
    if fundo:
        dff = dff[dff["FUNDO"].isin(fundo)]
    if etapa:
        dff = dff[dff["ETAPA"].isin(etapa)]
    if campo:
        dff = dff[dff["CAMPO"].isin(campo)]
    if turno:
        dff = dff[dff["TURNO"].isin(turno)]
    if variedad:
        dff = dff[dff["VARIEDAD"].isin(variedad)]
    if edad_final:
        dff = dff[dff["EDAD PLANTA FINAL"].isin(edad_final)]

    dff = dff[(dff["SEMANA"] >= semana_min) & (dff["SEMANA"] <= semana_max)]
    return dff

def _sort_campaign_categories(campaign_series: pd.Series):
    uniq = campaign_series.dropna().astype(str).unique().tolist()
    def to_int_or_big(x):
        try:
            return int(str(x).strip())
        except Exception:
            return 10**9
    uniq_sorted = sorted(uniq, key=lambda x: (to_int_or_big(x), str(x)))
    return uniq_sorted

def campaign_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "CAMPAÑA", "KG", "KG/HA", "PESO BAYA (g)", "CALIBRE BAYA (mm)", "ÁREA EJECUTADA (Ha COSECHADA)"
        ])

    out = []
    for camp, g in df.groupby("CAMPAÑA", dropna=False):
        out.append({
            "CAMPAÑA": str(camp),
            "KG": float(pd.to_numeric(g["kilogramos"], errors="coerce").sum(skipna=True)),
            "KG/HA": weighted_mean(g["KG/HA"], g[W_COL]),
            "PESO BAYA (g)": weighted_mean(g["PESO BAYA (g)"], g[W_COL]),
            "CALIBRE BAYA (mm)": weighted_mean(g["CALIBRE BAYA (mm)"], g[W_COL]),
            "ÁREA EJECUTADA (Ha COSECHADA)": float(pd.to_numeric(g[W_COL], errors="coerce").sum(skipna=True)),
        })

    res = pd.DataFrame(out)
    cats = _sort_campaign_categories(res["CAMPAÑA"])
    res["CAMPAÑA"] = pd.Categorical(res["CAMPAÑA"], categories=cats, ordered=True)
    return res.sort_values("CAMPAÑA").reset_index(drop=True)

def aggregate_level(df: pd.DataFrame, level_cols: list, y_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=level_cols + ["y_pond", "w_sum", "kg_sum"])

    rows = []
    for keys, g in df.groupby(level_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: keys[i] for i, col in enumerate(level_cols)}
        rec["y_pond"] = weighted_mean(g[y_col], g[W_COL])
        rec["w_sum"] = float(pd.to_numeric(g[W_COL], errors="coerce").sum(skipna=True))
        rec["kg_sum"] = float(pd.to_numeric(g["kilogramos"], errors="coerce").sum(skipna=True))
        rows.append(rec)
    return pd.DataFrame(rows)

def best_worst_turno_by_campaign_variety(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    level = ["CAMPAÑA", "VARIEDAD", "ETAPA", "CAMPO", "TURNO"]
    agg = aggregate_level(df, level, "KG/HA").rename(columns={"y_pond": "KG/HA_pond"})

    for _, col in STRUCT_COLS.items():
        tmp = aggregate_level(df, level, col)[level + ["y_pond"]].rename(columns={"y_pond": f"{col}_pond"})
        agg = agg.merge(tmp, on=level, how="left")

    out_rows = []
    for (camp, var), g in agg.groupby(["CAMPAÑA", "VARIEDAD"], dropna=False):
        g2 = g.dropna(subset=["KG/HA_pond"]).copy()
        if g2.empty:
            continue
        g2 = g2.sort_values("KG/HA_pond", ascending=False)
        best = g2.iloc[0]
        worst = g2.iloc[-1]

        out_rows.append({
            "CAMPAÑA": str(camp),
            "VARIEDAD": var,

            "TURNO_MAX": best["TURNO"],
            "ETAPA_MAX": best["ETAPA"],
            "CAMPO_MAX": best["CAMPO"],
            "KG/HA_MAX (pond)": best["KG/HA_pond"],
            "MADERAS (pond)": best.get("MADERAS PRINCIPALES_pond", np.nan),
            "CORTES (pond)": best.get("CORTES_pond", np.nan),
            "BROTES TOTALES (pond)": best.get("BROTES TOTALES_pond", np.nan),

            "TURNO_MIN": worst["TURNO"],
            "ETAPA_MIN": worst["ETAPA"],
            "CAMPO_MIN": worst["CAMPO"],
            "KG/HA_MIN (pond)": worst["KG/HA_pond"],
        })

    return pd.DataFrame(out_rows), agg

def corr_heatmap(df: pd.DataFrame) -> go.Figure:
    dd = df.copy()
    use = [c for c in CORR_COLS if c in dd.columns]
    dd = dd[use]

    for c in use:
        dd[c] = to_numeric_safe(dd[c])

    keep = [c for c in use if dd[c].notna().sum() >= 20]
    dd = dd[keep]
    if dd.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="No hay suficientes columnas numéricas con data para correlación.", showarrow=False)
        return fig

    corr = dd.corr(numeric_only=True)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorbar=dict(title="corr"),
        )
    )
    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Mapa de correlaciones (solo columnas solicitadas)",
    )
    return fig

def compute_campaign_axis_start_week(dff: pd.DataFrame) -> int:
    if dff.empty:
        return 1

    d = dff.copy()
    d["AÑO_NUM"] = to_numeric_safe(d["AÑO"])
    d["CAMPAÑA_NUM"] = pd.to_numeric(d["CAMPAÑA"].astype(str).str.strip(), errors="coerce")

    starts = []
    for camp, g in d.groupby("CAMPAÑA", dropna=False):
        camp_num = pd.to_numeric(str(camp).strip(), errors="coerce")

        if pd.notna(camp_num):
            h = g[g["AÑO_NUM"] == camp_num].dropna(subset=["SEMANA"])
            if not h.empty:
                starts.append(int(h["SEMANA"].min()))
                continue

        g2 = g.dropna(subset=["SEMANA"])
        if not g2.empty:
            starts.append(int(g2["SEMANA"].min()))

    if not starts:
        return int(d["SEMANA"].min()) if d["SEMANA"].notna().any() else 1

    start_global = int(min(starts))
    start_global = max(1, min(52, start_global))
    return start_global

def first_valid(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if not s.empty else np.nan

def compute_kg_planta_campaign(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["CAMPAÑA", "KG/PLANTA_calc"])

    unit_cols = ["CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]

    rows = []
    for camp, g in df.groupby("CAMPAÑA", dropna=False):
        kg_total = float(pd.to_numeric(g["kilogramos"], errors="coerce").sum(skipna=True))

        base_turno = (
            g.groupby(unit_cols, dropna=False)
             .agg(
                 Ha_TURNO_u=("Ha TURNO", first_valid),
                 DENSIDAD_u=("DENSIDAD", first_valid)
             )
             .reset_index()
        )

        base_turno["Ha_TURNO_u"] = to_numeric_safe(base_turno["Ha_TURNO_u"])
        base_turno["DENSIDAD_u"] = to_numeric_safe(base_turno["DENSIDAD_u"])
        base_turno["den_turno"] = base_turno["Ha_TURNO_u"] * base_turno["DENSIDAD_u"]

        den_total = float(base_turno["den_turno"].sum(skipna=True))
        kg_planta = (kg_total / den_total) if den_total > 0 else np.nan

        rows.append({
            "CAMPAÑA": str(camp),
            "KG_TOTAL": kg_total,
            "DEN_TOTAL_HA_TURNO_X_DENSIDAD": den_total,
            "KG/PLANTA_calc": kg_planta
        })

    out = pd.DataFrame(rows)
    out["CAMPAÑA"] = pd.Categorical(out["CAMPAÑA"], categories=_sort_campaign_categories(out["CAMPAÑA"]), ordered=True)
    return out.sort_values("CAMPAÑA").reset_index(drop=True)

# --------------------------
# HELPERS: INDICE DE VIGOR
# --------------------------
def minmax_normalize(series: pd.Series, global_min: float, global_max: float) -> pd.Series:
    s = to_numeric_safe(series)
    if pd.isna(global_min) or pd.isna(global_max) or global_max == global_min:
        return pd.Series(np.nan, index=s.index)
    return (s - global_min) / (global_max - global_min)

def build_vigor_unit_table(df_full: pd.DataFrame, dff_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una tabla única por CAMPAÑA-ETAPA-CAMPO-TURNO-VARIEDAD
    usando:
    - variables estructurales: último valor válido (porque es evaluación final de campaña)
    - KG/HA: promedio ponderado por Ha COSECHADA sobre la unidad filtrada
    """
    if dff_filtered.empty:
        return pd.DataFrame()

    # Base filtrada a nivel unidad productiva
    agg_dict = {col: (col, first_valid) for col in VIGOR_VARS if col in dff_filtered.columns}
    agg_dict.update({
        "EDAD PLANTA FINAL": ("EDAD PLANTA FINAL", first_valid),
        "TIPO PODA": ("TIPO PODA", first_valid),
        "KG/HA_pond": ("KG/HA", lambda s: np.nan),   # placeholder
        "AREA_sum": ("Ha COSECHADA", "sum"),
    })

    unit = (
        dff_filtered.groupby(UNIT_COLS_VIGOR, dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    # KG/HA ponderado por unidad
    kg_rows = []
    for keys, g in dff_filtered.groupby(UNIT_COLS_VIGOR, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {UNIT_COLS_VIGOR[i]: keys[i] for i in range(len(UNIT_COLS_VIGOR))}
        rec["KG/HA_pond"] = weighted_mean(g["KG/HA"], g[W_COL])
        kg_rows.append(rec)
    kg_df = pd.DataFrame(kg_rows)

    unit = unit.drop(columns=["KG/HA_pond"], errors="ignore").merge(kg_df, on=UNIT_COLS_VIGOR, how="left")

    # Min y max globales usando TODA la base para mantener comparabilidad
    global_stats = {}
    for col in VIGOR_VARS:
        if col in df_full.columns:
            s = to_numeric_safe(df_full[col]).dropna()
            global_stats[col] = {
                "min": float(s.min()) if not s.empty else np.nan,
                "max": float(s.max()) if not s.empty else np.nan,
            }

    # Normalizar cada variable
    norm_cols = []
    for col in VIGOR_VARS:
        if col in unit.columns and col in global_stats:
            ncol = f"{col}_NORM"
            unit[ncol] = minmax_normalize(unit[col], global_stats[col]["min"], global_stats[col]["max"])
            norm_cols.append(ncol)

    # Índice básico = promedio simple de variables normalizadas disponibles
    if norm_cols:
        unit["N_VARS_VIGOR"] = unit[norm_cols].notna().sum(axis=1)
        unit["INDICE_VIGOR"] = unit[norm_cols].mean(axis=1, skipna=True)
        unit.loc[unit["N_VARS_VIGOR"] == 0, "INDICE_VIGOR"] = np.nan
    else:
        unit["N_VARS_VIGOR"] = 0
        unit["INDICE_VIGOR"] = np.nan

    return unit

def vigor_summary_by_campaign(unit_vigor: pd.DataFrame) -> pd.DataFrame:
    if unit_vigor.empty:
        return pd.DataFrame(columns=["CAMPAÑA", "INDICE_VIGOR", "KG/HA_pond"])

    rows = []
    for camp, g in unit_vigor.groupby("CAMPAÑA", dropna=False):
        rows.append({
            "CAMPAÑA": str(camp),
            "INDICE_VIGOR": float(pd.to_numeric(g["INDICE_VIGOR"], errors="coerce").mean(skipna=True)),
            "KG/HA_pond": float(pd.to_numeric(g["KG/HA_pond"], errors="coerce").mean(skipna=True)),
            "N_UNIDADES": int(g["INDICE_VIGOR"].notna().sum()),
        })
    out = pd.DataFrame(rows)
    order = _sort_campaign_categories(out["CAMPAÑA"])
    out["CAMPAÑA"] = pd.Categorical(out["CAMPAÑA"], categories=order, ordered=True)
    return out.sort_values("CAMPAÑA").reset_index(drop=True)

# --------------------------
# UI: HEADER
# --------------------------
st.title("🫐 Fenología y estructura vs rendimiento (KG/HA) | Campañas 2022–2025")

# --------------------------
# LOAD
# --------------------------
if not os.path.exists(DATA_FILE):
    st.error(
        f"No encuentro el archivo **{DATA_FILE}** en la carpeta del app.\n\n"
        "✅ Solución:\n"
        "- Asegúrate que el Excel esté en el repo en la misma carpeta que `app.py`.\n"
        f"- Que el nombre sea EXACTO: `{DATA_FILE}`\n"
        f"- Y que la hoja se llame exactamente: `{REQ_SHEET}`"
    )
    st.stop()

df_raw = read_excel_path(DATA_FILE, REQ_SHEET)

missing = validate_cols(df_raw)
if missing:
    st.error("Faltan columnas requeridas en tu hoja DATA:")
    st.write(missing)
    st.stop()

df = df_raw.copy()

df["SEMANA"] = to_numeric_safe(df["SEMANA"]).fillna(0).astype(int)
df["CAMPAÑA"] = df["CAMPAÑA"].astype(str).str.strip()

num_main = [
    "AÑO", "kilogramos", "FLORES", "FRUTO CUAJADO", "FRUTO VERDE", "TOTAL DE FRUTOS",
    "Ha COSECHADA", "Ha TURNO", "KG/HA", "DENSIDAD", "FRUTO MADURO", "FRUTO ROSADO", "FRUTO CREMOSO",
    "PESO BAYA (g)", "PESO BAYA CREMOSO (g)", "CALIBRE BAYA (mm)", "CALIBRE CREMOSO (mm)",
    "SEMANA DE SIEMBRA",
    "MADERAS PRINCIPALES", "CORTES", "BROTES TOTALES", "TERMINALES", "EDAD PLANTA",
    "BP_N_BROTES_ULT", "BP_LONG_ULT", "BP_DIAM_ULT",
    "BS_N_BROTES_ULT", "BS_LONG_ULT", "BS_DIAM_ULT",
    "BT_N_BROTES_ULT", "BT_LONG_ULT", "BT_DIAM_ULT",
    "ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT"
]
for c in num_main:
    if c in df.columns:
        df[c] = to_numeric_safe(df[c])

df = ensure_categories_age(df)

# --------------------------
# FILTERS
# --------------------------
with st.sidebar:
    st.header("🎛️ Filtros")

    def ms(col):
        vals = sorted([v for v in df[col].dropna().unique().tolist()])
        return st.multiselect(col, vals, default=[])

    camp_f = ms("CAMPAÑA")
    fundo_f = ms("FUNDO")
    etapa_f = ms("ETAPA")
    campo_f = ms("CAMPO")
    turno_f = ms("TURNO")
    variedad_f = ms("VARIEDAD")
    edad_final_f = ms("EDAD PLANTA FINAL")

    sem_min, sem_max = int(df["SEMANA"].min()), int(df["SEMANA"].max())
    smin, smax = st.slider("SEMANA (rango)", sem_min, sem_max, (sem_min, sem_max))

dff = apply_filters(df, camp_f, fundo_f, etapa_f, campo_f, turno_f, variedad_f, edad_final_f, smin, smax)

# --------------------------
# RESUMEN POR CAMPAÑA
# --------------------------
st.subheader("Resumen por campaña (ponderado por Ha COSECHADA)")

res_camp = campaign_summary(dff)
st.dataframe(
    res_camp.style.format({
        "KG": "{:,.2f}",
        "KG/HA": "{:,.2f}",
        "PESO BAYA (g)": "{:,.2f}",
        "CALIBRE BAYA (mm)": "{:,.2f}",
        "ÁREA EJECUTADA (Ha COSECHADA)": "{:,.2f}",
    }),
    use_container_width=True
)

# --------------------------
# SCATTER: X vs MÉTRICA Y (ponderada) + BEST/WORST
# --------------------------
st.subheader("Dispersión (X vs métrica ponderada) + BEST/WORST")

left, right = st.columns([0.28, 0.72])

with left:
    y_label = st.selectbox("Métrica Y (ponderada por Ha COSECHADA)", list(METRIC_Y_OPTIONS.keys()), index=0)
    y_col = METRIC_Y_OPTIONS[y_label]

    numeric_candidates = []
    for c in dff.columns:
        if c in [y_col, W_COL]:
            continue
        if pd.api.types.is_numeric_dtype(dff[c]):
            numeric_candidates.append(c)

    preferred = [
        "FLORES", "FRUTO CUAJADO", "FRUTO VERDE", "TOTAL DE FRUTOS", "DENSIDAD", "FRUTO MADURO",
        "FRUTO ROSADO", "FRUTO CREMOSO",
        "MADERAS PRINCIPALES", "CORTES", "BROTES TOTALES", "TERMINALES", "EDAD PLANTA", "SEMANA DE SIEMBRA",
        "BP_N_BROTES_ULT", "BP_LONG_ULT", "BP_DIAM_ULT",
        "BS_N_BROTES_ULT", "BS_LONG_ULT", "BS_DIAM_ULT",
        "BT_N_BROTES_ULT", "BT_LONG_ULT", "BT_DIAM_ULT",
        "ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT"
    ]
    ordered = [c for c in preferred if c in numeric_candidates] + [c for c in sorted(numeric_candidates) if c not in preferred]

    x_col = st.selectbox("Variable X", ordered, index=0 if ordered else 0)

with right:
    level = ["TURNO"]
    agg_sc = aggregate_level(dff, level, y_col).rename(columns={"y_pond": "Y_pond"})

    if x_col:
        tmpx = aggregate_level(dff, level, x_col)[["TURNO", "y_pond"]].rename(columns={"y_pond": "X_pond"})
        agg_sc = agg_sc.merge(tmpx, on="TURNO", how="left")

    agg_sc["POINT"] = "NORMAL"
    if not agg_sc.empty and agg_sc["Y_pond"].notna().any():
        idx_best = agg_sc["Y_pond"].idxmax()
        idx_worst = agg_sc["Y_pond"].idxmin()
        agg_sc.loc[idx_best, "POINT"] = "BEST"
        agg_sc.loc[idx_worst, "POINT"] = "WORST"

    title_sc = f"{x_col} vs {y_label} | Nivel: TURNO"
    fig_sc = px.scatter(
        agg_sc,
        x="X_pond" if "X_pond" in agg_sc.columns else None,
        y="Y_pond",
        hover_data=["TURNO", "w_sum", "kg_sum", "POINT"],
        color="POINT",
        title=title_sc,
    )
    fig_sc.update_layout(xaxis_title=x_col, yaxis_title=f"{y_label} (ponderado)")
    st.plotly_chart(fig_sc, use_container_width=True)

st.divider()

# --------------------------
# CURVA SEMANAL: KG/HA ponderado por Ha COSECHADA
# --------------------------
st.subheader("Curva semanal de KG/HA (comparación por campaña)")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    rows = []
    for (camp, sem), g in dff.groupby(["CAMPAÑA", "SEMANA"], dropna=False):
        rows.append({
            "CAMPAÑA": str(camp),
            "SEMANA": int(sem),
            "KG/HA_pond": weighted_mean(g["KG/HA"], g[W_COL]),
        })
    wk = pd.DataFrame(rows)

    start_week = compute_campaign_axis_start_week(dff)
    wk["SEMANA_EJE"] = np.where(wk["SEMANA"] < start_week, wk["SEMANA"] + 52, wk["SEMANA"]).astype(int)
    wk = wk.sort_values(["CAMPAÑA", "SEMANA_EJE"])

    max_eje = int(wk["SEMANA_EJE"].max())
    tickvals = list(range(start_week, 53))
    if max_eje >= 53:
        tickvals += list(range(53, max_eje + 1))
    ticktext = [str(v) if v <= 52 else str(v - 52) for v in tickvals]

    fig_wk = px.line(
        wk, x="SEMANA_EJE", y="KG/HA_pond", color="CAMPAÑA",
        markers=True,
        title=f"Promedio KG/HA por semana (ponderado por Ha COSECHADA) | Eje continuo desde semana {start_week}"
    )
    fig_wk.update_layout(
        xaxis=dict(
            title="SEMANA (orden real por campaña)",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        ),
        yaxis=dict(title="KG/HA_pond"),
    )
    st.plotly_chart(fig_wk, use_container_width=True)

st.divider()

# --------------------------
# BOXPLOTS: SIEMBRA y EDAD PLANTA FINAL
# --------------------------
st.subheader("KG/HA ponderado: Boxplot por SIEMBRA y por EDAD PLANTA FINAL")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    turn_level = ["CAMPAÑA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD", "SIEMBRA", "EDAD PLANTA FINAL"]
    agg_turn = aggregate_level(dff, turn_level, "KG/HA").rename(columns={"y_pond": "KG/HA_pond"})
    agg_turn = agg_turn.dropna(subset=["KG/HA_pond"])

    b1, b2 = st.columns(2)

    with b1:
        fig_siem = px.box(
            agg_turn,
            x="SIEMBRA",
            y="KG/HA_pond",
            points="outliers",
            title="KG/HA ponderado por SIEMBRA (boxplot)"
        )
        fig_siem.update_layout(xaxis=dict(type="category"))
        st.plotly_chart(fig_siem, use_container_width=True)

    with b2:
        agg_turn["EDAD PLANTA FINAL"] = agg_turn["EDAD PLANTA FINAL"].astype(str)
        order_age = ["1", "2", "3+"]
        fig_age = px.box(
            agg_turn,
            x="EDAD PLANTA FINAL",
            y="KG/HA_pond",
            category_orders={"EDAD PLANTA FINAL": order_age},
            points="outliers",
            title="KG/HA ponderado por EDAD PLANTA FINAL (boxplot)"
        )
        fig_age.update_layout(xaxis=dict(type="category"))
        st.plotly_chart(fig_age, use_container_width=True)

st.divider()

# --------------------------
# KG/PLANTA con fórmula
# --------------------------
st.subheader("KG/PLANTA calculado vs campañas")

kgp = compute_kg_planta_campaign(dff)
if kgp.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    fig_kp = px.line(
        kgp,
        x="CAMPAÑA",
        y="KG/PLANTA_calc",
        markers=True,
        title="KG/PLANTA calculado vs CAMPAÑA | Fórmula: KG total / SUMA(Ha TURNO × DENSIDAD por turno único)"
    )
    fig_kp.update_layout(xaxis=dict(type="category"), yaxis=dict(title="KG/PLANTA_calc"))
    st.plotly_chart(fig_kp, use_container_width=True)

st.divider()

# --------------------------
# VARIEDADES: ranking ponderado + heatmap VS campañas
# --------------------------
st.subheader("Variedades: ranking (KG/HA ponderado) + VS por campañas")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    top_n = st.slider("Top N variedades (por frecuencia)", 5, 25, 10)

    level_var = ["VARIEDAD", "CAMPAÑA"]
    agg_v = aggregate_level(dff, level_var, "KG/HA").rename(columns={"y_pond": "KG/HA_pond"})
    agg_v["CAMPAÑA"] = agg_v["CAMPAÑA"].astype(str)

    freq = (
        dff.groupby("VARIEDAD")["TURNO"]
        .count()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"TURNO": "n"})
    )

    rows = []
    for var, g in dff.groupby("VARIEDAD", dropna=False):
        rows.append({"VARIEDAD": var, "KG/HA_pond": weighted_mean(g["KG/HA"], g[W_COL])})
    avg_var = pd.DataFrame(rows).merge(freq, on="VARIEDAD", how="left").fillna({"n": 0})
    avg_var = avg_var.sort_values("n", ascending=False).head(top_n)

    fig_rank = px.bar(
        avg_var.sort_values("KG/HA_pond", ascending=True),
        x="KG/HA_pond", y="VARIEDAD",
        orientation="h",
        title="Promedio KG/HA ponderado (Top variedades por frecuencia)"
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    top_vars = avg_var["VARIEDAD"].tolist()
    hm = agg_v[agg_v["VARIEDAD"].isin(top_vars)].copy()
    pivot = hm.pivot_table(index="VARIEDAD", columns="CAMPAÑA", values="KG/HA_pond", aggfunc="mean")
    pivot = pivot.reindex(index=top_vars)

    fig_hm = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=pivot.index.tolist(),
            colorbar=dict(title="avg KG/HA_pond"),
        )
    )
    fig_hm.update_layout(
        title="VS: VARIEDAD x CAMPAÑA (KG/HA ponderado)",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(type="category")
    )
    st.plotly_chart(fig_hm, use_container_width=True)

st.divider()

# --------------------------
# BEST vs WORST TURNO dentro de (CAMPAÑA + VARIEDAD)
# --------------------------
st.subheader("Best vs Worst TURNO dentro de (CAMPAÑA + VARIEDAD)")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    vars_available = sorted(dff["VARIEDAD"].dropna().unique().tolist())
    var_pick = st.selectbox("Selecciona VARIEDAD", vars_available, index=0 if vars_available else 0)

    d_var = dff[dff["VARIEDAD"] == var_pick].copy()

    bw_table, turno_level = best_worst_turno_by_campaign_variety(d_var)

    if bw_table.empty:
        st.info("No se pudo calcular best/worst con los datos actuales.")
    else:
        st.dataframe(
            bw_table.style.format({
                "KG/HA_MAX (pond)": "{:,.4f}",
                "KG/HA_MIN (pond)": "{:,.4f}",
                "MADERAS (pond)": "{:,.4f}",
                "CORTES (pond)": "{:,.4f}",
                "BROTES TOTALES (pond)": "{:,.4f}",
            }),
            use_container_width=True
        )

        camps = sorted(bw_table["CAMPAÑA"].unique().tolist())
        camp_pick = st.selectbox("Campaña para gráfico", camps, index=len(camps)-1 if camps else 0)

        row = bw_table[bw_table["CAMPAÑA"] == camp_pick].iloc[0]

        cats = [
            f"MAX | {row['TURNO_MAX']} ({row['ETAPA_MAX']}-{row['CAMPO_MAX']})",
            f"MIN | {row['TURNO_MIN']} ({row['ETAPA_MIN']}-{row['CAMPO_MIN']})",
        ]
        kgvals = [row["KG/HA_MAX (pond)"], row["KG/HA_MIN (pond)"]]

        tl = turno_level.copy()
        tl["CAMPAÑA"] = tl["CAMPAÑA"].astype(str)

        rec_best = tl[
            (tl["CAMPAÑA"] == camp_pick) &
            (tl["VARIEDAD"] == var_pick) &
            (tl["TURNO"] == row["TURNO_MAX"]) &
            (tl["ETAPA"] == row["ETAPA_MAX"]) &
            (tl["CAMPO"] == row["CAMPO_MAX"])
        ].head(1)

        rec_worst = tl[
            (tl["CAMPAÑA"] == camp_pick) &
            (tl["VARIEDAD"] == var_pick) &
            (tl["TURNO"] == row["TURNO_MIN"]) &
            (tl["ETAPA"] == row["ETAPA_MIN"]) &
            (tl["CAMPO"] == row["CAMPO_MIN"])
        ].head(1)

        maderas = [
            rec_best["MADERAS PRINCIPALES_pond"].iloc[0] if not rec_best.empty else np.nan,
            rec_worst["MADERAS PRINCIPALES_pond"].iloc[0] if not rec_worst.empty else np.nan,
        ]
        cortes = [
            rec_best["CORTES_pond"].iloc[0] if not rec_best.empty else np.nan,
            rec_worst["CORTES_pond"].iloc[0] if not rec_worst.empty else np.nan,
        ]
        brotes = [
            rec_best["BROTES TOTALES_pond"].iloc[0] if not rec_best.empty else np.nan,
            rec_worst["BROTES TOTALES_pond"].iloc[0] if not rec_worst.empty else np.nan,
        ]

        fig_bw = go.Figure()
        fig_bw.add_trace(go.Bar(x=cats, y=kgvals, name="KG/HA (ponderado)", yaxis="y1"))
        fig_bw.add_trace(go.Scatter(x=cats, y=maderas, mode="lines+markers", name="MADERAS PRINCIPALES (pond)", yaxis="y2"))
        fig_bw.add_trace(go.Scatter(x=cats, y=cortes, mode="lines+markers", name="CORTES (pond)", yaxis="y2"))
        fig_bw.add_trace(go.Scatter(x=cats, y=brotes, mode="lines+markers", name="BROTES TOTALES (pond)", yaxis="y2"))

        fig_bw.update_layout(
            title=f"KG/HA MAX vs MIN (por TURNO) + Estructura | CAMPAÑA {camp_pick} | VARIEDAD {var_pick}",
            xaxis=dict(type="category"),
            yaxis=dict(title="KG/HA (ponderado)"),
            yaxis2=dict(title="Estructura (pond)", overlaying="y", side="right"),
            legend=dict(orientation="h"),
            height=520
        )
        st.plotly_chart(fig_bw, use_container_width=True)

st.divider()

# --------------------------
# NUEVA VISTA 1: KG/HA por EDAD y CAMPAÑA
# --------------------------
st.subheader("KG/HA ponderado por edad y campaña")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    rows = []
    for (camp, edad), g in dff.groupby(["CAMPAÑA", "EDAD PLANTA FINAL"], dropna=False):
        rows.append({
            "CAMPAÑA": str(camp),
            "EDAD PLANTA FINAL": str(edad),
            "KG/HA_pond": weighted_mean(g["KG/HA"], g[W_COL])
        })

    age_camp = pd.DataFrame(rows)
    campaign_order = _sort_campaign_categories(age_camp["CAMPAÑA"])
    order_age = ["1", "2", "3+"]
    age_camp["CAMPAÑA"] = pd.Categorical(age_camp["CAMPAÑA"], categories=campaign_order, ordered=True)
    age_camp["EDAD PLANTA FINAL"] = pd.Categorical(age_camp["EDAD PLANTA FINAL"], categories=order_age, ordered=True)
    age_camp = age_camp.sort_values(["CAMPAÑA", "EDAD PLANTA FINAL"])

    fig_agecamp = px.bar(
        age_camp,
        x="CAMPAÑA",
        y="KG/HA_pond",
        color="EDAD PLANTA FINAL",
        barmode="group",
        category_orders={"CAMPAÑA": campaign_order, "EDAD PLANTA FINAL": order_age},
        title="KG/HA ponderado por EDAD PLANTA FINAL y CAMPAÑA"
    )
    fig_agecamp.update_layout(
        xaxis=dict(type="category", categoryorder="array", categoryarray=campaign_order),
        yaxis=dict(title="KG/HA_pond")
    )
    st.plotly_chart(fig_agecamp, use_container_width=True)

st.divider()

# --------------------------
# NUEVA VISTA 2: DENSIDAD vs RENDIMIENTO
# --------------------------
st.subheader("Densidad vs rendimiento")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    level_den = ["CAMPAÑA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]
    agg_y = aggregate_level(dff, level_den, "KG/HA").rename(columns={"y_pond": "KG/HA_pond"})
    agg_x = aggregate_level(dff, level_den, "DENSIDAD")[level_den + ["y_pond"]].rename(columns={"y_pond": "DENSIDAD_pond"})
    den_df = agg_y.merge(agg_x, on=level_den, how="left")

    fig_den = px.scatter(
        den_df,
        x="DENSIDAD_pond",
        y="KG/HA_pond",
        color="CAMPAÑA",
        hover_data=["ETAPA", "CAMPO", "TURNO", "VARIEDAD", "w_sum", "kg_sum"],
        title="DENSIDAD vs KG/HA ponderado"
    )
    fig_den.update_layout(
        xaxis_title="DENSIDAD",
        yaxis_title="KG/HA_pond"
    )
    st.plotly_chart(fig_den, use_container_width=True)

st.divider()

# --------------------------
# CORRELACIONES
# --------------------------
st.subheader("Mapa de correlaciones (solo columnas seleccionadas)")
fig_corr = corr_heatmap(dff)
st.plotly_chart(fig_corr, use_container_width=True)

st.divider()

# --------------------------
# INDICE DE VIGOR VEGETATIVO POST-PODA
# --------------------------
st.subheader("Índice de vigor vegetativo post-poda")

unit_vigor = build_vigor_unit_table(df, dff)

if unit_vigor.empty or unit_vigor["INDICE_VIGOR"].notna().sum() == 0:
    st.warning("No hay suficientes datos estructurales para calcular el índice de vigor con los filtros actuales.")
else:
    # 1) Tabla resumen por unidad productiva
    st.markdown("### Resumen del índice por unidad productiva")
    show_cols = [
        "CAMPAÑA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
        "EDAD PLANTA FINAL", "TIPO PODA", "INDICE_VIGOR", "KG/HA_pond", "N_VARS_VIGOR"
    ]
    show_cols = [c for c in show_cols if c in unit_vigor.columns]
    st.dataframe(
        unit_vigor[show_cols].sort_values(["CAMPAÑA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]).style.format({
            "INDICE_VIGOR": "{:,.4f}",
            "KG/HA_pond": "{:,.2f}",
        }),
        use_container_width=True
    )

    st.divider()

    # 2) Índice agregado por campaña
    st.markdown("### Índice de vigor por campaña")
    vig_camp = vigor_summary_by_campaign(unit_vigor)

    fig_vig_camp = go.Figure()
    fig_vig_camp.add_trace(go.Bar(
        x=vig_camp["CAMPAÑA"].astype(str),
        y=vig_camp["INDICE_VIGOR"],
        name="Índice de vigor"
    ))
    fig_vig_camp.add_trace(go.Scatter(
        x=vig_camp["CAMPAÑA"].astype(str),
        y=vig_camp["KG/HA_pond"],
        mode="lines+markers",
        name="KG/HA promedio",
        yaxis="y2"
    ))
    fig_vig_camp.update_layout(
        title="Índice de vigor y KG/HA promedio por campaña",
        xaxis=dict(type="category"),
        yaxis=dict(title="Índice de vigor"),
        yaxis2=dict(title="KG/HA", overlaying="y", side="right"),
        height=450
    )
    st.plotly_chart(fig_vig_camp, use_container_width=True)

    st.divider()

    # 3) Índice por edad de planta
    st.markdown("### Índice de vigor por edad de planta")
    age_order = ["1", "2", "3+"]
    age_vig = (
        unit_vigor.assign(**{"EDAD PLANTA FINAL": unit_vigor["EDAD PLANTA FINAL"].astype(str)})
        .groupby("EDAD PLANTA FINAL", dropna=False)
        .agg(
            INDICE_VIGOR=("INDICE_VIGOR", "mean"),
            KG_HA_PROM=("KG/HA_pond", "mean"),
            N=("INDICE_VIGOR", "count")
        )
        .reset_index()
    )
    age_vig["EDAD PLANTA FINAL"] = pd.Categorical(age_vig["EDAD PLANTA FINAL"], categories=age_order, ordered=True)
    age_vig = age_vig.sort_values("EDAD PLANTA FINAL")

    fig_age_vig = go.Figure()
    fig_age_vig.add_trace(go.Bar(
        x=age_vig["EDAD PLANTA FINAL"].astype(str),
        y=age_vig["INDICE_VIGOR"],
        name="Índice de vigor"
    ))
    fig_age_vig.add_trace(go.Scatter(
        x=age_vig["EDAD PLANTA FINAL"].astype(str),
        y=age_vig["KG_HA_PROM"],
        mode="lines+markers",
        name="KG/HA promedio",
        yaxis="y2"
    ))
    fig_age_vig.update_layout(
        title="Índice de vigor y KG/HA promedio por edad de planta",
        xaxis=dict(type="category"),
        yaxis=dict(title="Índice de vigor"),
        yaxis2=dict(title="KG/HA", overlaying="y", side="right"),
        height=450
    )
    st.plotly_chart(fig_age_vig, use_container_width=True)

    st.divider()

    # 4) Índice por tipo de poda
    st.markdown("### Índice de vigor por tipo de poda")
    poda_vig = (
        unit_vigor.groupby("TIPO PODA", dropna=False)
        .agg(
            INDICE_VIGOR=("INDICE_VIGOR", "mean"),
            KG_HA_PROM=("KG/HA_pond", "mean"),
            N=("INDICE_VIGOR", "count")
        )
        .reset_index()
    )

    if not poda_vig.empty:
        fig_poda_vig = go.Figure()
        fig_poda_vig.add_trace(go.Bar(
            x=poda_vig["TIPO PODA"].astype(str),
            y=poda_vig["INDICE_VIGOR"],
            name="Índice de vigor"
        ))
        fig_poda_vig.add_trace(go.Scatter(
            x=poda_vig["TIPO PODA"].astype(str),
            y=poda_vig["KG_HA_PROM"],
            mode="lines+markers",
            name="KG/HA promedio",
            yaxis="y2"
        ))
        fig_poda_vig.update_layout(
            title="Índice de vigor y KG/HA promedio por tipo de poda",
            xaxis=dict(type="category"),
            yaxis=dict(title="Índice de vigor"),
            yaxis2=dict(title="KG/HA", overlaying="y", side="right"),
            height=450
        )
        st.plotly_chart(fig_poda_vig, use_container_width=True)

    st.divider()

    # 5) Relación índice de vigor vs KG/HA
    st.markdown("### Relación entre índice de vigor y KG/HA")
    fig_vig_scatter = px.scatter(
        unit_vigor,
        x="INDICE_VIGOR",
        y="KG/HA_pond",
        color="CAMPAÑA",
        hover_data=["ETAPA", "CAMPO", "TURNO", "VARIEDAD", "EDAD PLANTA FINAL", "TIPO PODA"],
        title="Índice de vigor vs KG/HA por unidad productiva"
    )
    fig_vig_scatter.update_layout(
        xaxis_title="Índice de vigor vegetativo",
        yaxis_title="KG/HA ponderado"
    )
    st.plotly_chart(fig_vig_scatter, use_container_width=True)

    st.divider()

    # 6) Composición del índice (promedio de variables normalizadas)
    st.markdown("### Composición del índice de vigor (promedio de variables normalizadas)")
    comp_cols = [f"{c}_NORM" for c in VIGOR_VARS if f"{c}_NORM" in unit_vigor.columns]
    if comp_cols:
        comp_means = unit_vigor[comp_cols].mean(skipna=True).reset_index()
        comp_means.columns = ["VARIABLE", "PROM_NORM"]
        comp_means["VARIABLE"] = comp_means["VARIABLE"].str.replace("_NORM", "", regex=False)

        fig_comp = px.bar(
            comp_means.sort_values("PROM_NORM", ascending=True),
            x="PROM_NORM",
            y="VARIABLE",
            orientation="h",
            title="Promedio normalizado de variables estructurales usadas en el índice"
        )
        fig_comp.update_layout(
            xaxis_title="Promedio normalizado",
            yaxis_title="Variable"
        )
        st.plotly_chart(fig_comp, use_container_width=True)
