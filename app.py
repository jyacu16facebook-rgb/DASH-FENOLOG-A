# app.py
# ==========================================================
# DASH: Fenología y estructura vs rendimiento (KG/HA)
# - Carga AUTOMÁTICA desde archivo local en repo (xlsx) | Hoja: DATA
# - Filtros: Fundo, Etapa, Campo, Turno, Variedad, Semana, Campaña, EDAD PLANTA FINAL
# - Métricas:
#   * KG = SUMA(kilogramos)  (sin ponderar)
#   * KG/HA resumen campaña = SUMA(kilogramos) / SUMA(Ha COSECHADA)
#   * Scatter:
#       - si Y = KG/HA => promedio simple por unidad
#       - si Y = PESO / CALIBRE => ponderado por Ha COSECHADA
#   * Área ejecutada = SUMA(Ha COSECHADA)
# - Boxplots:
#   * KG/HA simple por SIEMBRA FINAL
#   * KG/HA simple por EDAD PLANTA FINAL
#   * PESO BAYA (g) simple por SIEMBRA FINAL
#   * vista biométrica dinámica por SIEMBRA FINAL
# - Curva semanal: KG/HA promedio simple (EJE X continuo por campaña)
# - Variedades: ranking simple + heatmap VS por campaña
# - Vista adicional: KG/PLANTA con fórmula:
#       KG/PLANTA = kilogramos / (Ha TURNO * DENSIDAD)
#   respetando el TURNO como unidad única para evitar duplicar área por semana
# - Vistas agregadas:
#   (1) KG/HA por edad y campaña
#   (2) Índice de vigor vegetativo post-poda
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
    "SIEMBRA", "SIEMBRA FINAL"
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

BIOMETRIC_COLS = [
    "MADERAS PRINCIPALES", "CORTES", "BROTES TOTALES", "TERMINALES",
    "EDAD PLANTA", "EDAD PLANTA FINAL",
    "BP_N_BROTES_ULT", "BP_LONG_ULT", "BP_DIAM_ULT",
    "BS_N_BROTES_ULT", "BS_LONG_ULT", "BS_DIAM_ULT",
    "BT_N_BROTES_ULT", "BT_LONG_ULT", "BT_DIAM_ULT",
    "ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT"
]

# --------------------------
# HELPERS
# --------------------------
def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def simple_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() == 0:
        return np.nan
    return float(x.mean(skipna=True))

def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    mask = x.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))

def ratio_sum_over_sum(num: pd.Series, den: pd.Series) -> float:
    num_sum = float(pd.to_numeric(num, errors="coerce").sum(skipna=True))
    den_sum = float(pd.to_numeric(den, errors="coerce").sum(skipna=True))
    if den_sum <= 0:
        return np.nan
    return num_sum / den_sum

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
            "KG/HA": ratio_sum_over_sum(g["kilogramos"], g["Ha COSECHADA"]),
            "PESO BAYA (g)": weighted_mean(g["PESO BAYA (g)"], g[W_COL]),
            "CALIBRE BAYA (mm)": weighted_mean(g["CALIBRE BAYA (mm)"], g[W_COL]),
            "ÁREA EJECUTADA (Ha COSECHADA)": float(pd.to_numeric(g[W_COL], errors="coerce").sum(skipna=True)),
        })

    res = pd.DataFrame(out)
    cats = _sort_campaign_categories(res["CAMPAÑA"])
    res["CAMPAÑA"] = pd.Categorical(res["CAMPAÑA"], categories=cats, ordered=True)
    return res.sort_values("CAMPAÑA").reset_index(drop=True)

def aggregate_level(df: pd.DataFrame, level_cols: list, y_col: str, mode: str = "weighted") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=level_cols + ["y_val", "w_sum", "kg_sum"])

    rows = []
    for keys, g in df.groupby(level_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: keys[i] for i, col in enumerate(level_cols)}

        if mode == "simple":
            rec["y_val"] = simple_mean(g[y_col])
        elif mode == "ratio_kg_area":
            rec["y_val"] = ratio_sum_over_sum(g["kilogramos"], g["Ha COSECHADA"])
        else:
            rec["y_val"] = weighted_mean(g[y_col], g[W_COL])

        rec["w_sum"] = float(pd.to_numeric(g[W_COL], errors="coerce").sum(skipna=True))
        rec["kg_sum"] = float(pd.to_numeric(g["kilogramos"], errors="coerce").sum(skipna=True))
        rows.append(rec)
    return pd.DataFrame(rows)

def best_worst_turno_by_campaign_variety(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    level = ["CAMPAÑA", "VARIEDAD", "ETAPA", "CAMPO", "TURNO"]
    agg = aggregate_level(df, level, "KG/HA", mode="weighted").rename(columns={"y_val": "KG/HA_pond"})

    for _, col in STRUCT_COLS.items():
        tmp = aggregate_level(df, level, col, mode="weighted")[level + ["y_val"]].rename(columns={"y_val": f"{col}_pond"})
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

def build_siembra_final_biometric_summary(dff: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    cols_needed = ["SIEMBRA FINAL", metric_col]
    if dff.empty or any(c not in dff.columns for c in cols_needed):
        return pd.DataFrame()

    tmp = dff[cols_needed].copy()
    tmp["SIEMBRA FINAL"] = tmp["SIEMBRA FINAL"].astype(str).str.strip().str.upper()
    tmp = tmp[tmp["SIEMBRA FINAL"].isin(["SUELO", "MACETA"])]
    tmp[metric_col] = to_numeric_safe(tmp[metric_col])
    tmp = tmp.dropna(subset=[metric_col])

    if tmp.empty:
        return pd.DataFrame()

    summary = (
        tmp.groupby("SIEMBRA FINAL")[metric_col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(columns={
            "count": "N",
            "mean": "PROMEDIO",
            "median": "MEDIANA",
            "std": "DESV_STD",
            "min": "MIN",
            "max": "MAX"
        })
    )

    prom_suelo = summary.loc[summary["SIEMBRA FINAL"] == "SUELO", "PROMEDIO"]
    prom_maceta = summary.loc[summary["SIEMBRA FINAL"] == "MACETA", "PROMEDIO"]
    delta = np.nan
    if not prom_suelo.empty and not prom_maceta.empty:
        delta = float(prom_suelo.iloc[0] - prom_maceta.iloc[0])

    summary["DELTA_SUELO_MENOS_MACETA"] = delta
    return summary

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
    - variables estructurales: último valor válido
    - KG/HA: promedio ponderado por Ha COSECHADA sobre la unidad filtrada
    """
    if dff_filtered.empty:
        return pd.DataFrame()

        # Base filtrada a nivel unidad productiva
    agg_dict = {col: (col, first_valid) for col in VIGOR_VARS if col in dff_filtered.columns}
    agg_dict.update({
        "EDAD PLANTA FINAL": ("EDAD PLANTA FINAL", first_valid),
        "TIPO PODA": ("TIPO PODA", first_valid),
        "KG/HA_pond": ("KG/HA", lambda s: np.nan),
        "AREA_sum": ("Ha COSECHADA", "sum"),
    })

    unit = (
        dff_filtered.groupby(UNIT_COLS_VIGOR, dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    kg_rows = []
    for keys, g in dff_filtered.groupby(UNIT_COLS_VIGOR, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {UNIT_COLS_VIGOR[i]: keys[i] for i in range(len(UNIT_COLS_VIGOR))}
        rec["KG/HA_pond"] = weighted_mean(g["KG/HA"], g[W_COL])
        kg_rows.append(rec)
    kg_df = pd.DataFrame(kg_rows)

    unit = unit.drop(columns=["KG/HA_pond"], errors="ignore").merge(kg_df, on=UNIT_COLS_VIGOR, how="left")

    global_stats = {}
    for col in VIGOR_VARS:
        if col in df_full.columns:
            s = to_numeric_safe(df_full[col]).dropna()
            global_stats[col] = {
                "min": float(s.min()) if not s.empty else np.nan,
                "max": float(s.max()) if not s.empty else np.nan,
            }

    norm_cols = []
    for col in VIGOR_VARS:
        if col in unit.columns and col in global_stats:
            ncol = f"{col}_NORM"
            unit[ncol] = minmax_normalize(unit[col], global_stats[col]["min"], global_stats[col]["max"])
            norm_cols.append(ncol)

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
st.subheader("Resumen por campaña (KG/HA = KG total / área ejecutada total)")

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
# SCATTER: X vs MÉTRICA por campaña (un solo gráfico)
# --------------------------
st.subheader("Dispersión (X vs métrica) por campaña")

left, right = st.columns([0.28, 0.72])

with left:
    y_label = st.selectbox("Métrica Y", list(METRIC_Y_OPTIONS.keys()), index=0)
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
    if dff.empty or not x_col:
        st.warning("No hay datos con los filtros actuales.")
    else:
        level = ["CAMPAÑA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]

        if y_col == "KG/HA":
            y_mode = "ratio_kg_area"
            y_title = "KG/HA (KG total / área ejecutada)"
        else:
            y_mode = "weighted"
            y_title = f"{y_label} (ponderado)"

        if x_col == "KG/HA":
            x_mode = "ratio_kg_area"
        else:
            x_mode = "simple" if x_col in ["FLORES", "FRUTO CUAJADO", "FRUTO VERDE", "TOTAL DE FRUTOS",
                                           "FRUTO MADURO", "FRUTO ROSADO", "FRUTO CREMOSO",
                                           "MADERAS PRINCIPALES", "CORTES", "BROTES TOTALES", "TERMINALES",
                                           "EDAD PLANTA", "SEMANA DE SIEMBRA",
                                           "BP_N_BROTES_ULT", "BP_LONG_ULT", "BP_DIAM_ULT",
                                           "BS_N_BROTES_ULT", "BS_LONG_ULT", "BS_DIAM_ULT",
                                           "BT_N_BROTES_ULT", "BT_LONG_ULT", "BT_DIAM_ULT",
                                           "ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT"] else "weighted"

        agg_sc = aggregate_level(dff, level, y_col, mode=y_mode).rename(columns={"y_val": "Y_val"})
        tmpx = aggregate_level(dff, level, x_col, mode=x_mode)[level + ["y_val"]].rename(columns={"y_val": "X_val"})
        agg_sc = agg_sc.merge(tmpx, on=level, how="left")

        agg_sc["CAMPAÑA"] = agg_sc["CAMPAÑA"].astype(str)
        agg_sc["POINT"] = "NORMAL"

        if not agg_sc.empty and agg_sc["Y_val"].notna().any():
            for camp, g in agg_sc.groupby("CAMPAÑA", dropna=False):
                g2 = g.dropna(subset=["Y_val"])
                if g2.empty:
                    continue
                idx_best = g2["Y_val"].idxmax()
                idx_worst = g2["Y_val"].idxmin()
                agg_sc.loc[idx_best, "POINT"] = "BEST"
                agg_sc.loc[idx_worst, "POINT"] = "WORST"

        fig_sc = px.scatter(
            agg_sc,
            x="X_val",
            y="Y_val",
            color="CAMPAÑA",
            symbol="POINT",
            hover_data=["CAMPAÑA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD", "w_sum", "kg_sum", "POINT"],
            title=f"{x_col} vs {y_label} | Nivel: unidad productiva"
        )

        fig_sc.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_title,
            legend_title="CAMPAÑA",
        )
        st.plotly_chart(fig_sc, use_container_width=True)

st.divider()

# --------------------------
# CURVA SEMANAL: KG/HA promedio simple
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
            "KG/HA_simple": simple_mean(g["KG/HA"]),
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
        wk, x="SEMANA_EJE", y="KG/HA_simple", color="CAMPAÑA",
        markers=True,
        title=f"Promedio KG/HA por semana (promedio simple) | Eje continuo desde semana {start_week}"
    )
    fig_wk.update_layout(
        xaxis=dict(
            title="SEMANA (orden real por campaña)",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        ),
        yaxis=dict(title="KG/HA promedio simple"),
    )
    st.plotly_chart(fig_wk, use_container_width=True)

st.divider()

# --------------------------
# BOXPLOTS: SIEMBRA FINAL y EDAD PLANTA FINAL
# --------------------------
st.subheader("KG/HA simple: Boxplot por SIEMBRA FINAL y por EDAD PLANTA FINAL")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    turn_level = ["CAMPAÑA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD", "SIEMBRA FINAL", "EDAD PLANTA FINAL"]

    agg_turn_kgha = aggregate_level(dff, turn_level, "KG/HA", mode="simple").rename(columns={"y_val": "KG/HA_simple"})
    agg_turn_kgha = agg_turn_kgha.dropna(subset=["KG/HA_simple"])

    b1, b2 = st.columns(2)

    with b1:
        fig_siem = px.box(
            agg_turn_kgha,
            x="SIEMBRA FINAL",
            y="KG/HA_simple",
            points="outliers",
            title="KG/HA simple por SIEMBRA FINAL (boxplot)"
        )
        fig_siem.update_layout(xaxis=dict(type="category"))
        st.plotly_chart(fig_siem, use_container_width=True)

    with b2:
        agg_turn_kgha["EDAD PLANTA FINAL"] = agg_turn_kgha["EDAD PLANTA FINAL"].astype(str)
        order_age = ["1", "2", "3+"]
        fig_age = px.box(
            agg_turn_kgha,
            x="EDAD PLANTA FINAL",
            y="KG/HA_simple",
            category_orders={"EDAD PLANTA FINAL": order_age},
            points="outliers",
            title="KG/HA simple por EDAD PLANTA FINAL (boxplot)"
        )
        fig_age.update_layout(xaxis=dict(type="category"))
        st.plotly_chart(fig_age, use_container_width=True)

    st.divider()

    st.subheader("PESO BAYA (g): Boxplot por SIEMBRA FINAL")
    agg_turn_peso = aggregate_level(dff, turn_level, "PESO BAYA (g)", mode="simple").rename(columns={"y_val": "PESO_BAYA_simple"})
    agg_turn_peso = agg_turn_peso.dropna(subset=["PESO_BAYA_simple"])

    fig_peso_siem = px.box(
        agg_turn_peso,
        x="SIEMBRA FINAL",
        y="PESO_BAYA_simple",
        points="outliers",
        title="PESO BAYA (g) promedio simple por SIEMBRA FINAL (boxplot)"
    )
    fig_peso_siem.update_layout(
        xaxis=dict(type="category"),
        yaxis=dict(title="PESO BAYA (g) promedio simple")
    )
    st.plotly_chart(fig_peso_siem, use_container_width=True)

    st.divider()

    st.subheader("Diferencias biométricas entre SIEMBRA FINAL")
    st.caption("Compara SUELO vs MACETA para una variable biométrica a elección.")

    biom_col_pick = st.selectbox(
        "Selecciona variable biométrica",
        BIOMETRIC_COLS,
        index=BIOMETRIC_COLS.index("ALTURA_PLANTA_ULT") if "ALTURA_PLANTA_ULT" in BIOMETRIC_COLS else 0
    )

    bio_left, bio_right = st.columns([0.68, 0.32])

    with bio_left:
        bio_df = dff[["SIEMBRA FINAL", biom_col_pick]].copy()
        bio_df["SIEMBRA FINAL"] = bio_df["SIEMBRA FINAL"].astype(str).str.strip().str.upper()
        bio_df[biom_col_pick] = to_numeric_safe(bio_df[biom_col_pick])
        bio_df = bio_df[bio_df["SIEMBRA FINAL"].isin(["SUELO", "MACETA"])]
        bio_df = bio_df.dropna(subset=[biom_col_pick])

        if bio_df.empty:
            st.info("No hay datos suficientes para la vista biométrica con los filtros actuales.")
        else:
            fig_bio = px.box(
                bio_df,
                x="SIEMBRA FINAL",
                y=biom_col_pick,
                points="outliers",
                title=f"{biom_col_pick} por SIEMBRA FINAL"
            )
            fig_bio.update_layout(
                xaxis=dict(type="category"),
                yaxis=dict(title=biom_col_pick)
            )
            st.plotly_chart(fig_bio, use_container_width=True)

    with bio_right:
        bio_summary = build_siembra_final_biometric_summary(dff, biom_col_pick)
        if bio_summary.empty:
            st.info("No hay resumen disponible.")
        else:
            st.dataframe(
                bio_summary.style.format({
                    "PROMEDIO": "{:,.2f}",
                    "MEDIANA": "{:,.2f}",
                    "DESV_STD": "{:,.2f}",
                    "MIN": "{:,.2f}",
                    "MAX": "{:,.2f}",
                    "DELTA_SUELO_MENOS_MACETA": "{:,.2f}",
                }),
                use_container_width=True
            )

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
# VARIEDADES: ranking simple + heatmap VS campañas
# --------------------------
st.subheader("Variedades: ranking (KG/HA simple) + VS por campañas")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    top_n = st.slider("Top N variedades (por frecuencia)", 5, 25, 10)

    level_var = ["VARIEDAD", "CAMPAÑA"]
    agg_v = aggregate_level(dff, level_var, "KG/HA", mode="simple").rename(columns={"y_val": "KG/HA_simple"})
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
        rows.append({"VARIEDAD": var, "KG/HA_simple": simple_mean(g["KG/HA"])})
    avg_var = pd.DataFrame(rows).merge(freq, on="VARIEDAD", how="left").fillna({"n": 0})
    avg_var = avg_var.sort_values("n", ascending=False).head(top_n)

    fig_rank = px.bar(
        avg_var.sort_values("KG/HA_simple", ascending=True),
        x="KG/HA_simple", y="VARIEDAD",
        orientation="h",
        title="Promedio KG/HA simple (Top variedades por frecuencia)"
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    top_vars = avg_var["VARIEDAD"].tolist()
    hm = agg_v[agg_v["VARIEDAD"].isin(top_vars)].copy()
    pivot = hm.pivot_table(index="VARIEDAD", columns="CAMPAÑA", values="KG/HA_simple", aggfunc="mean")
    pivot = pivot.reindex(index=top_vars)

    fig_hm = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=pivot.index.tolist(),
            colorbar=dict(title="avg KG/HA_simple"),
        )
    )
    fig_hm.update_layout(
        title="VS: VARIEDAD x CAMPAÑA (KG/HA simple)",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(type="category")
    )
    st.plotly_chart(fig_hm, use_container_width=True)

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
