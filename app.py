# app.py
# ==========================================================
# DASH: Fenología vs rendimiento
# ==========================================================

import os
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, shapiro, levene, f_oneway, kruskal

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
    "SIEMBRA", "SIEMBRA FINAL",
    "SEG DENSIDAD"
]

W_COL = "Ha COSECHADA"

# Unidad productiva única
UNIT_COLS_BASE = ["CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]

METRIC_Y_OPTIONS = {
    "KG/HA": "KG/HA",
    "KG/PLANTA": "KG/PLANTA",
    "PESO BAYA (g)": "PESO BAYA (g)",
    "CALIBRE BAYA (mm)": "CALIBRE BAYA (mm)",
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

BIOMETRIC_COLS = [
    "MADERAS PRINCIPALES", "CORTES", "BROTES TOTALES", "TERMINALES",
    "EDAD PLANTA", "EDAD PLANTA FINAL",
    "BP_N_BROTES_ULT", "BP_LONG_ULT", "BP_DIAM_ULT",
    "BS_N_BROTES_ULT", "BS_LONG_ULT", "BS_DIAM_ULT",
    "BT_N_BROTES_ULT", "BT_LONG_ULT", "BT_DIAM_ULT",
    "ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT"
]

SUM_X_COLS = [
    "FLORES", "FRUTO CUAJADO", "FRUTO VERDE", "TOTAL DE FRUTOS",
    "FRUTO MADURO", "FRUTO ROSADO", "FRUTO CREMOSO"
]

# Variables comparativas tipo suma
SUM_COMPARATIVE_COLS = SUM_X_COLS + ["kilogramos"]

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

def sum_numeric(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").sum(skipna=True))

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

def first_valid(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if not s.empty else np.nan

def build_unique_turno_table(df_subset: pd.DataFrame) -> pd.DataFrame:
    if df_subset.empty:
        return pd.DataFrame(columns=UNIT_COLS_BASE + ["Ha_TURNO_u", "DENSIDAD_u"])

    base = (
        df_subset.groupby(UNIT_COLS_BASE, dropna=False)
        .agg(
            Ha_TURNO_u=("Ha TURNO", first_valid),
            DENSIDAD_u=("DENSIDAD", first_valid),
        )
        .reset_index()
    )

    base["Ha_TURNO_u"] = to_numeric_safe(base["Ha_TURNO_u"])
    base["DENSIDAD_u"] = to_numeric_safe(base["DENSIDAD_u"])
    return base

def unique_turno_area_sum(df_subset: pd.DataFrame) -> float:
    base = build_unique_turno_table(df_subset)
    if base.empty:
        return 0.0
    return float(base["Ha_TURNO_u"].sum(skipna=True))

def unique_turno_plants_sum(df_subset: pd.DataFrame) -> float:
    base = build_unique_turno_table(df_subset)
    if base.empty:
        return 0.0
    base["PLANTAS_EST"] = base["Ha_TURNO_u"] * base["DENSIDAD_u"]
    return float(base["PLANTAS_EST"].sum(skipna=True))

def ratio_kg_over_unique_turno_area(df_subset: pd.DataFrame) -> float:
    kg_sum = sum_numeric(df_subset["kilogramos"])
    area_sum = unique_turno_area_sum(df_subset)
    if area_sum <= 0:
        return np.nan
    return kg_sum / area_sum

def ratio_kg_planta_over_unique_turno(df_subset: pd.DataFrame) -> float:
    kg_sum = sum_numeric(df_subset["kilogramos"])
    plantas_sum = unique_turno_plants_sum(df_subset)
    if plantas_sum <= 0:
        return np.nan
    return kg_sum / plantas_sum

def weighted_density_by_turno_area(df_subset: pd.DataFrame) -> float:
    base = build_unique_turno_table(df_subset)
    if base.empty:
        return np.nan
    return weighted_mean(base["DENSIDAD_u"], base["Ha_TURNO_u"])

def campaign_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "CAMPAÑA", "KG", "KG/HA", "PESO", "CALIBRE", "ÁREA"
        ])

    out = []
    for camp, g in df.groupby("CAMPAÑA", dropna=False):
        out.append({
            "CAMPAÑA": str(camp),
            "KG": sum_numeric(g["kilogramos"]),
            "KG/HA": ratio_kg_over_unique_turno_area(g),
            "PESO": weighted_mean(g["PESO BAYA (g)"], g[W_COL]),
            "CALIBRE": weighted_mean(g["CALIBRE BAYA (mm)"], g[W_COL]),
            "ÁREA": unique_turno_area_sum(g),
        })

    res = pd.DataFrame(out)
    cats = _sort_campaign_categories(res["CAMPAÑA"])
    res["CAMPAÑA"] = pd.Categorical(res["CAMPAÑA"], categories=cats, ordered=True)
    return res.sort_values("CAMPAÑA").reset_index(drop=True)

def aggregate_level(df: pd.DataFrame, level_cols: list, y_col: str, mode: str = "weighted") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=level_cols + ["y_val", "w_sum", "kg_sum", "area_sum"])

    rows = []
    for keys, g in df.groupby(level_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: keys[i] for i, col in enumerate(level_cols)}

        if mode == "simple":
            rec["y_val"] = simple_mean(g[y_col])
        elif mode == "sum":
            rec["y_val"] = sum_numeric(g[y_col])
        elif mode == "weighted":
            rec["y_val"] = weighted_mean(g[y_col], g[W_COL])
        elif mode == "weighted_kg":
            rec["y_val"] = weighted_mean(g[y_col], g["kilogramos"])
        elif mode == "ratio_kg_area_turno":
            rec["y_val"] = ratio_kg_over_unique_turno_area(g)
        elif mode == "ratio_kg_planta_turno":
            rec["y_val"] = ratio_kg_planta_over_unique_turno(g)
        else:
            rec["y_val"] = np.nan

        rec["w_sum"] = sum_numeric(g[W_COL])
        rec["kg_sum"] = sum_numeric(g["kilogramos"])
        rec["area_sum"] = unique_turno_area_sum(g)
        rows.append(rec)

    return pd.DataFrame(rows)

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
        fig.add_annotation(text="Sin datos suficientes.", showarrow=False)
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
        title="Correlaciones",
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

def compute_kg_planta_campaign(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["CAMPAÑA", "KG/PLANTA"])

    rows = []
    for camp, g in df.groupby("CAMPAÑA", dropna=False):
        kg_total = sum_numeric(g["kilogramos"])
        area_total = unique_turno_area_sum(g)
        densidad_pond = weighted_density_by_turno_area(g)
        plantas_total = unique_turno_plants_sum(g)
        kg_planta = (kg_total / plantas_total) if plantas_total > 0 else np.nan

        rows.append({
            "CAMPAÑA": str(camp),
            "KG_TOTAL": kg_total,
            "ÁREA": area_total,
            "DENSIDAD": densidad_pond,
            "PLANTAS": plantas_total,
            "KG/PLANTA": kg_planta
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

    summary["DELTA"] = delta
    return summary

def compute_full_dynamic_axis_range(series: pd.Series, lower_zero: bool = True):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None

    real_min = float(s.min())
    real_max = float(s.max())

    if real_min == real_max:
        pad = max(abs(real_max) * 0.10, 1)
        low = real_min - pad
        high = real_max + pad
    else:
        pad = (real_max - real_min) * 0.08
        low = real_min - pad
        high = real_max + pad

    if lower_zero and high > 0:
        low = max(0, low)

    if high <= low:
        high = low + 1

    return [low, high]

def compute_pearson_stats(x: pd.Series, y: pd.Series):
    tmp = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    n = len(tmp)

    if n < 3:
        return np.nan, np.nan, n

    if tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return np.nan, np.nan, n

    r, p = pearsonr(tmp["x"], tmp["y"])
    return float(r), float(p), int(n)

def analyze_variance_by_group(df_plot: pd.DataFrame, group_col: str, value_col: str) -> dict:
    out = {
        "prueba": "NA",
        "estadistico_nombre": "NA",
        "estadistico": np.nan,
        "p_valor": np.nan,
        "n": 0,
        "grupos": 0,
        "normalidad_ok": False,
        "homocedasticidad_ok": False,
    }

    if df_plot.empty or group_col not in df_plot.columns or value_col not in df_plot.columns:
        return out

    tmp = df_plot[[group_col, value_col]].copy()
    tmp[group_col] = tmp[group_col].astype(str).str.strip()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.replace({group_col: {"nan": np.nan, "None": np.nan, "": np.nan}})
    tmp = tmp.dropna(subset=[group_col, value_col]).copy()

    if tmp.empty:
        return out

    grouped = []
    for gname, gdf in tmp.groupby(group_col, dropna=False):
        vals = pd.to_numeric(gdf[value_col], errors="coerce").dropna().values
        if len(vals) > 0:
            grouped.append(vals)

    out["n"] = int(sum(len(v) for v in grouped))
    out["grupos"] = int(len(grouped))

    if len(grouped) < 2:
        return out

    normality_flags = []
    for vals in grouped:
        if len(vals) < 3 or len(np.unique(vals)) < 2:
            normality_flags.append(False)
            continue

        vals_test = vals
        if len(vals_test) > 5000:
            vals_test = np.random.default_rng(42).choice(vals_test, size=5000, replace=False)

        try:
            _, p_norm = shapiro(vals_test)
            normality_flags.append(bool(p_norm > 0.05))
        except Exception:
            normality_flags.append(False)

    normalidad_ok = all(normality_flags) if normality_flags else False
    out["normalidad_ok"] = normalidad_ok

    homocedasticidad_ok = False
    if all(len(vals) >= 2 for vals in grouped):
        try:
            _, p_lev = levene(*grouped, center="median")
            homocedasticidad_ok = bool(p_lev > 0.05)
        except Exception:
            homocedasticidad_ok = False

    out["homocedasticidad_ok"] = homocedasticidad_ok

    if normalidad_ok and homocedasticidad_ok and all(len(vals) >= 2 for vals in grouped):
        try:
            stat, pval = f_oneway(*grouped)
            out["prueba"] = "ANOVA"
            out["estadistico_nombre"] = "F"
            out["estadistico"] = float(stat)
            out["p_valor"] = float(pval)
            return out
        except Exception:
            pass

    try:
        stat, pval = kruskal(*grouped)
        out["prueba"] = "Kruskal-Wallis"
        out["estadistico_nombre"] = "H"
        out["estadistico"] = float(stat)
        out["p_valor"] = float(pval)
        return out
    except Exception:
        return out

def format_p_value_decimal(p):
    if pd.isna(p):
        return "NA"
    if p < 0.000001:
        return "0.000001"
    return f"{p:.6f}"

def render_variance_metrics(result: dict):
    c1, c2, c3, c4, c5 = st.columns(5)

    def card(label, value):
        st.markdown(
            f"""
            <div style="padding-top:2px; padding-bottom:2px;">
                <div style="font-size:12px; color:#555; margin-bottom:2px; white-space:nowrap;">{label}</div>
                <div style="font-size:16px; font-weight:600; line-height:1.15; word-break:break-word;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c1:
        card("Prueba", result.get("prueba", "NA"))

    with c2:
        est_name = result.get("estadistico_nombre", "NA")
        est_val = result.get("estadistico", np.nan)
        if pd.notna(est_val):
            card(f"Estadístico ({est_name})", f"{est_val:.4f}")
        else:
            card(f"Estadístico ({est_name})", "NA")

    with c3:
        pval = result.get("p_valor", np.nan)
        card("p-valor", format_p_value_decimal(pval))

    with c4:
        card("N", f"{int(result.get('n', 0)):,}")

    with c5:
        card("Grupos", f"{int(result.get('grupos', 0))}")

def build_group_descriptive_summary(df_plot: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    if df_plot.empty or group_col not in df_plot.columns or value_col not in df_plot.columns:
        return pd.DataFrame(columns=["GRUPO", "MEDIA", "DESV_STD", "CV(%)"])

    tmp = df_plot[[group_col, value_col]].copy()
    tmp[group_col] = tmp[group_col].astype(str).str.strip()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.replace({group_col: {"nan": np.nan, "None": np.nan, "": np.nan}})
    tmp = tmp.dropna(subset=[group_col, value_col]).copy()

    if tmp.empty:
        return pd.DataFrame(columns=["GRUPO", "MEDIA", "DESV_STD", "CV(%)"])

    desc = (
        tmp.groupby(group_col, dropna=False)[value_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            group_col: "GRUPO",
            "mean": "MEDIA",
            "std": "DESV_STD"
        })
    )

    desc["DESV_STD"] = desc["DESV_STD"].fillna(0)

    desc["CV_VAL"] = np.where(
        desc["MEDIA"].abs() > 0,
        desc["DESV_STD"] / desc["MEDIA"],
        np.nan
    )

    desc["CV(%)"] = desc["CV_VAL"].apply(
        lambda x: "NA" if pd.isna(x) else f"{x*100:.0f}%"
    )

    desc = desc.sort_values("MEDIA", ascending=False).reset_index(drop=True)

    return desc[["GRUPO", "MEDIA", "DESV_STD", "CV(%)"]]

def render_group_descriptive_summary(df_plot: pd.DataFrame, group_col: str, value_col: str):
    desc = build_group_descriptive_summary(df_plot, group_col, value_col)
    if desc.empty:
        st.info("No hay resumen descriptivo por grupo.")
        return

    st.dataframe(
        desc.style.format({
            "MEDIA": "{:,.4f}",
            "DESV_STD": "{:,.4f}",
        }),
        use_container_width=True
    )

# --------------------------
# HELPERS NUEVOS: MAX / MIN
# --------------------------
def get_comparative_candidates(df_input: pd.DataFrame) -> list:
    candidates = []
    exclude = {"SEMANA"}
    for c in df_input.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df_input[c]):
            candidates.append(c)

    preferred = [
        "kilogramos", "KG/HA", "PESO BAYA (g)", "CALIBRE BAYA (mm)",
        "FLORES", "FRUTO CUAJADO", "FRUTO VERDE", "TOTAL DE FRUTOS",
        "FRUTO MADURO", "FRUTO ROSADO", "FRUTO CREMOSO",
        "Ha TURNO", "Ha COSECHADA", "DENSIDAD",
        "MADERAS PRINCIPALES", "CORTES", "BROTES TOTALES", "TERMINALES",
        "EDAD PLANTA", "SEMANA DE SIEMBRA",
        "BP_N_BROTES_ULT", "BP_LONG_ULT", "BP_DIAM_ULT",
        "BS_N_BROTES_ULT", "BS_LONG_ULT", "BS_DIAM_ULT",
        "BT_N_BROTES_ULT", "BT_LONG_ULT", "BT_DIAM_ULT",
        "ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT"
    ]

    ordered = [c for c in preferred if c in candidates] + [c for c in sorted(candidates) if c not in preferred]
    return ordered

def comparative_value_turno(df_subset: pd.DataFrame, comp_var: str) -> float:
    if comp_var not in df_subset.columns:
        return np.nan

    if comp_var in SUM_COMPARATIVE_COLS:
        return sum_numeric(df_subset[comp_var])

    return simple_mean(df_subset[comp_var])

def comparative_value_campo(df_subset: pd.DataFrame, comp_var: str) -> float:
    if comp_var not in df_subset.columns:
        return np.nan

    if comp_var in SUM_COMPARATIVE_COLS:
        return sum_numeric(df_subset[comp_var])

    rows = []
    for keys, g in df_subset.groupby(UNIT_COLS_BASE, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ha_turno = to_numeric_safe(g["Ha TURNO"])
        ha_turno_u = first_valid(ha_turno)
        val_turno = simple_mean(g[comp_var])
        rows.append({
            "VAL": val_turno,
            "HA_TURNO": ha_turno_u
        })

    base = pd.DataFrame(rows)
    if base.empty:
        return np.nan

    return weighted_mean(base["VAL"], base["HA_TURNO"])

def build_entity_maxmin_summary(
    df_input: pd.DataFrame,
    level_mode: str,
    metric_name: str,
    comp_vars: list
):
    if df_input.empty:
        return pd.DataFrame(), pd.DataFrame()

    if level_mode == "TURNO":
        level_cols = UNIT_COLS_BASE
    else:
        level_cols = ["CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "VARIEDAD"]

    rows = []

    for keys, g in df_input.groupby(level_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        rec = {col: keys[i] for i, col in enumerate(level_cols)}

        if metric_name == "KG/HA":
            metric_val = ratio_kg_over_unique_turno_area(g)
        else:
            metric_val = weighted_mean(g["PESO BAYA (g)"], g["kilogramos"])

        for comp_var in comp_vars:
            if level_mode == "TURNO":
                rec[comp_var] = comparative_value_turno(g, comp_var)
            else:
                rec[comp_var] = comparative_value_campo(g, comp_var)

        rec["METRICA"] = metric_val
        rec["KG_TOTAL"] = sum_numeric(g["kilogramos"])
        rec["AREA_TURNO_UNICA"] = unique_turno_area_sum(g)
        rows.append(rec)

    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame()

    detail = detail.dropna(subset=["METRICA"]).copy()
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame()

    detail["CAMPAÑA"] = detail["CAMPAÑA"].astype(str)
    summary_rows = []

    for camp, gcamp in detail.groupby("CAMPAÑA", dropna=False):
        gcamp = gcamp.sort_values("METRICA", ascending=False).reset_index(drop=True)
        row_max = gcamp.iloc[0].copy()
        row_min = gcamp.iloc[-1].copy()

        if level_mode == "TURNO":
            for tipo, row in [("MAX", row_max), ("MIN", row_min)]:
                rec = {
                    "TIPO": tipo,
                    "CAMPAÑA": row["CAMPAÑA"],
                    "FUNDO": row["FUNDO"],
                    "ETAPA": row["ETAPA"],
                    "CAMPO": row["CAMPO"],
                    "TURNO": row["TURNO"],
                    "VARIEDAD": row["VARIEDAD"],
                    metric_name: row["METRICA"],
                    "KG_TOTAL": row["KG_TOTAL"],
                    "HA_TURNO_UNICA": row["AREA_TURNO_UNICA"],
                }
                for comp_var in comp_vars:
                    rec[comp_var] = row.get(comp_var, np.nan)
                summary_rows.append(rec)
        else:
            for tipo, row in [("MAX", row_max), ("MIN", row_min)]:
                rec = {
                    "TIPO": tipo,
                    "CAMPAÑA": row["CAMPAÑA"],
                    "FUNDO": row["FUNDO"],
                    "ETAPA": row["ETAPA"],
                    "CAMPO": row["CAMPO"],
                    "VARIEDAD": row["VARIEDAD"],
                    metric_name: row["METRICA"],
                    "KG_TOTAL": row["KG_TOTAL"],
                    "HA_TURNO_UNICA": row["AREA_TURNO_UNICA"],
                }
                for comp_var in comp_vars:
                    rec[comp_var] = row.get(comp_var, np.nan)
                summary_rows.append(rec)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return pd.DataFrame(), detail

    summary["CAMPAÑA"] = pd.Categorical(
        summary["CAMPAÑA"].astype(str),
        categories=_sort_campaign_categories(summary["CAMPAÑA"].astype(str)),
        ordered=True
    )
    summary = summary.sort_values(["CAMPAÑA", "TIPO"]).reset_index(drop=True)

    return summary, detail

def render_maxmin_chart(summary_df: pd.DataFrame, metric_name: str, comp_vars: list, level_mode: str):
    if summary_df.empty:
        st.info("No hay datos suficientes para mostrar la comparación MAX/MIN.")
        return

    df_plot = summary_df.copy()
    df_plot["CAMPAÑA"] = df_plot["CAMPAÑA"].astype(str)

    campaign_order = _sort_campaign_categories(df_plot["CAMPAÑA"])
    fig = go.Figure()

    color_map = {"MAX": "#156cc2", "MIN": "#8bbcf0"}

    for tipo in ["MAX", "MIN"]:
        d = df_plot[df_plot["TIPO"] == tipo].copy()
        d = d.set_index("CAMPAÑA").reindex(campaign_order).reset_index()

        customdata = []
        x_vals = []
        y_vals = []
        text_vals = []

        for _, r in d.iterrows():
            if pd.isna(r.get(metric_name, np.nan)):
                continue

            camp = r["CAMPAÑA"]
            if level_mode == "TURNO":
                label_entidad = f"{r['TURNO']} ({r['CAMPO']})"
            else:
                label_entidad = f"{r['CAMPO']}"

            x_vals.append(camp)
            y_vals.append(r[metric_name])
            text_vals.append(f"{r[metric_name]:,.0f}")
            customdata.append([label_entidad, tipo])

        fig.add_trace(go.Bar(
            x=x_vals,
            y=y_vals,
            name=f"{metric_name} {tipo}",
            yaxis="y1",
            width=0.25,
            offsetgroup=tipo,
            marker_color=color_map[tipo],
            text=text_vals,
            textposition="outside",
            textfont=dict(size=11),
            customdata=customdata,
            hovertemplate=(
                "Campaña: %{x}<br>"
                "Tipo: %{customdata[1]}<br>"
                f"{level_mode}: " + "%{customdata[0]}<br>"
                f"{metric_name}: %{{y:,.2f}}<extra></extra>"
            )
        ))

    for comp_var in comp_vars:
        x_vals = []
        y_vals = []
        text_vals = []

        for camp in campaign_order:
            for tipo in ["MAX", "MIN"]:
                row = df_plot[(df_plot["CAMPAÑA"] == camp) & (df_plot["TIPO"] == tipo)]
                if row.empty:
                    continue
                row = row.iloc[0]

                x_vals.append(camp)
                y_vals.append(row.get(comp_var, np.nan))
                text_vals.append("NA" if pd.isna(row.get(comp_var, np.nan)) else f"{row.get(comp_var):,.0f}")

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers+text",
            name=comp_var,
            yaxis="y2",
            text=text_vals,
            textposition="top center",
            textfont=dict(size=10),
            connectgaps=False
        ))

    fig.update_layout(
        title=f"MAX vs MIN de {metric_name} por {level_mode} + {', '.join(comp_vars)}",
        xaxis=dict(
            type="category",
            title="CAMPAÑA",
            categoryorder="array",
            categoryarray=campaign_order
        ),
        yaxis=dict(title=metric_name),
        yaxis2=dict(title="Variables comparativas", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        height=580,
        margin=dict(t=80, b=60),
        bargap=0.55,
        bargroupgap=0.15
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# UI: HEADER
# --------------------------
st.title("🫐 Fenología vs rendimiento | 2022–2025")

# --------------------------
# LOAD
# --------------------------
if not os.path.exists(DATA_FILE):
    st.error(
        f"No encuentro el archivo **{DATA_FILE}**.\n\n"
        "Verifica:\n"
        "- mismo folder que `app.py`\n"
        f"- nombre exacto: `{DATA_FILE}`\n"
        f"- hoja exacta: `{REQ_SHEET}`"
    )
    st.stop()

df_raw = read_excel_path(DATA_FILE, REQ_SHEET)

missing = validate_cols(df_raw)
if missing:
    st.error("Faltan columnas requeridas:")
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
    smin, smax = st.slider("SEMANA", sem_min, sem_max, (sem_min, sem_max))

dff = apply_filters(df, camp_f, fundo_f, etapa_f, campo_f, turno_f, variedad_f, edad_final_f, smin, smax)

# --------------------------
# RESUMEN
# --------------------------
st.subheader("Resumen por campaña")

res_camp = campaign_summary(dff)
st.dataframe(
    res_camp.style.format({
        "KG": "{:,.2f}",
        "KG/HA": "{:,.2f}",
        "PESO": "{:,.2f}",
        "CALIBRE": "{:,.2f}",
        "ÁREA": "{:,.2f}",
    }),
    use_container_width=True
)

# --------------------------
# SCATTER
# --------------------------
st.subheader("Dispersión")

left, right = st.columns([0.28, 0.72])

with left:
    y_label = st.selectbox("Métrica Y", list(METRIC_Y_OPTIONS.keys()), index=0)
    y_col = METRIC_Y_OPTIONS[y_label]

    numeric_candidates = []
    for c in dff.columns:
        if c in [W_COL]:
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
        level = UNIT_COLS_BASE

        if y_col == "KG/HA":
            y_mode = "ratio_kg_area_turno"
            y_title = "KG/HA"
        elif y_col == "KG/PLANTA":
            y_mode = "ratio_kg_planta_turno"
            y_title = "KG/PLANTA"
        else:
            y_mode = "weighted"
            y_title = y_label

        if x_col == "KG/HA":
            x_mode = "ratio_kg_area_turno"
        elif x_col in SUM_X_COLS:
            x_mode = "sum"
        elif x_col in [
            "MADERAS PRINCIPALES", "CORTES", "BROTES TOTALES", "TERMINALES",
            "EDAD PLANTA", "SEMANA DE SIEMBRA",
            "BP_N_BROTES_ULT", "BP_LONG_ULT", "BP_DIAM_ULT",
            "BS_N_BROTES_ULT", "BS_LONG_ULT", "BS_DIAM_ULT",
            "BT_N_BROTES_ULT", "BT_LONG_ULT", "BT_DIAM_ULT",
            "ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT"
        ]:
            x_mode = "simple"
        else:
            x_mode = "weighted"

        agg_sc = aggregate_level(dff, level, y_col, mode=y_mode).rename(columns={"y_val": "Y_val"})
        tmpx = aggregate_level(dff, level, x_col, mode=x_mode)[level + ["y_val"]].rename(columns={"y_val": "X_val"})
        agg_sc = agg_sc.merge(tmpx, on=level, how="left")

        agg_sc["CAMPAÑA"] = agg_sc["CAMPAÑA"].astype(str)
        agg_sc = agg_sc.dropna(subset=["X_val", "Y_val"]).copy()

        fig_sc = px.scatter(
            agg_sc,
            x="X_val",
            y="Y_val",
            color="CAMPAÑA",
            hover_data=["CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD", "kg_sum", "area_sum"],
            title=f"{x_col} vs {y_label}"
        )

        y_range = compute_full_dynamic_axis_range(agg_sc["Y_val"], lower_zero=True)

        fig_sc.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_title,
            legend_title="CAMPAÑA",
        )

        if y_range is not None:
            fig_sc.update_yaxes(range=y_range)

        st.plotly_chart(fig_sc, use_container_width=True)

        pearson_r, pearson_p, pearson_n = compute_pearson_stats(agg_sc["X_val"], agg_sc["Y_val"])
        m1, m2, m3 = st.columns(3)
        with m1:
            if pd.notna(pearson_r):
                st.metric("r de Pearson", f"{pearson_r:.4f}")
            else:
                st.metric("r de Pearson", "NA")
        with m2:
            st.metric("p-valor", format_p_value_decimal(pearson_p))
        with m3:
            st.metric("N", f"{pearson_n:,}")

st.divider()

# --------------------------
# CURVA SEMANAL
# --------------------------
st.subheader("Curva semanal")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    rows = []
    for (camp, sem), g in dff.groupby(["CAMPAÑA", "SEMANA"], dropna=False):
        rows.append({
            "CAMPAÑA": str(camp),
            "SEMANA": int(sem),
            "KG/HA": simple_mean(g["KG/HA"]),
        })
    wk = pd.DataFrame(rows)

    start_week = compute_campaign_axis_start_week(dff)
    wk["SEM_EJE"] = np.where(wk["SEMANA"] < start_week, wk["SEMANA"] + 52, wk["SEMANA"]).astype(int)
    wk = wk.sort_values(["CAMPAÑA", "SEM_EJE"])

    max_eje = int(wk["SEM_EJE"].max())
    tickvals = list(range(start_week, 53))
    if max_eje >= 53:
        tickvals += list(range(53, max_eje + 1))
    ticktext = [str(v) if v <= 52 else str(v - 52) for v in tickvals]

    fig_wk = px.line(
        wk, x="SEM_EJE", y="KG/HA", color="CAMPAÑA",
        markers=True,
        title="KG/HA por semana"
    )
    fig_wk.update_layout(
        xaxis=dict(
            title="SEMANA",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        ),
        yaxis=dict(title="KG/HA"),
    )
    st.plotly_chart(fig_wk, use_container_width=True)

st.divider()

# --------------------------
# BOXPLOTS
# --------------------------
st.subheader("Boxplots")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    turn_level = UNIT_COLS_BASE + ["SIEMBRA FINAL", "EDAD PLANTA FINAL"]

    agg_turn_kgha = aggregate_level(dff, turn_level, "KG/HA", mode="ratio_kg_area_turno").rename(columns={"y_val": "KG_HA"})
    agg_turn_kgha = agg_turn_kgha.dropna(subset=["KG_HA"])

    b1, b2 = st.columns(2)

    with b1:
        fig_siem = px.box(
            agg_turn_kgha,
            x="SIEMBRA FINAL",
            y="KG_HA",
            points="outliers",
            title="KG/HA por SIEMBRA"
        )
        fig_siem.update_layout(
            xaxis=dict(type="category", title="SIEMBRA"),
            yaxis=dict(title="KG/HA")
        )
        st.plotly_chart(fig_siem, use_container_width=True)

        anova_siem = analyze_variance_by_group(agg_turn_kgha, "SIEMBRA FINAL", "KG_HA")
        render_variance_metrics(anova_siem)
        render_group_descriptive_summary(agg_turn_kgha, "SIEMBRA FINAL", "KG_HA")

    with b2:
        agg_turn_kgha["EDAD PLANTA FINAL"] = agg_turn_kgha["EDAD PLANTA FINAL"].astype(str)
        order_age = ["1", "2", "3+"]
        fig_age = px.box(
            agg_turn_kgha,
            x="EDAD PLANTA FINAL",
            y="KG_HA",
            category_orders={"EDAD PLANTA FINAL": order_age},
            points="outliers",
            title="KG/HA por EDAD"
        )
        fig_age.update_layout(
            xaxis=dict(type="category", title="EDAD"),
            yaxis=dict(title="KG/HA")
        )
        st.plotly_chart(fig_age, use_container_width=True)

        anova_age = analyze_variance_by_group(agg_turn_kgha, "EDAD PLANTA FINAL", "KG_HA")
        render_variance_metrics(anova_age)
        render_group_descriptive_summary(agg_turn_kgha, "EDAD PLANTA FINAL", "KG_HA")

    st.divider()

    st.subheader("PESO BAYA")

    agg_turn_peso = aggregate_level(dff, turn_level, "PESO BAYA (g)", mode="weighted_kg").rename(columns={"y_val": "PESO"})
    agg_turn_peso = agg_turn_peso.dropna(subset=["PESO"])

    fig_peso_siem = px.box(
        agg_turn_peso,
        x="SIEMBRA FINAL",
        y="PESO",
        points="outliers",
        title="PESO por SIEMBRA"
    )
    fig_peso_siem.update_layout(
        xaxis=dict(type="category", title="SIEMBRA"),
        yaxis=dict(title="PESO")
    )
    st.plotly_chart(fig_peso_siem, use_container_width=True)

    anova_peso = analyze_variance_by_group(agg_turn_peso, "SIEMBRA FINAL", "PESO")
    render_variance_metrics(anova_peso)
    render_group_descriptive_summary(agg_turn_peso, "SIEMBRA FINAL", "PESO")

    st.divider()

    # --------------------------
    # NUEVO BOXPLOT: SEG DENSIDAD
    # --------------------------
    st.subheader("SEG DENSIDAD")
    st.caption("Compara SEG DENSIDAD con métrica dinámica: KG/HA o KG/PLANTA.")

    metric_seg_pick = st.selectbox(
        "Métrica Y para SEG DENSIDAD",
        ["KG/HA", "KG/PLANTA"],
        index=0
    )

    turn_level_seg = UNIT_COLS_BASE + ["SEG DENSIDAD"]

    if metric_seg_pick == "KG/HA":
        agg_turn_seg = aggregate_level(
            dff, turn_level_seg, "KG/HA", mode="ratio_kg_area_turno"
        ).rename(columns={"y_val": "METRIC_VAL"})
    else:
        agg_turn_seg = aggregate_level(
            dff, turn_level_seg, "KG/PLANTA", mode="ratio_kg_planta_turno"
        ).rename(columns={"y_val": "METRIC_VAL"})

    agg_turn_seg["SEG DENSIDAD"] = agg_turn_seg["SEG DENSIDAD"].astype(str).str.strip()
    agg_turn_seg = agg_turn_seg.replace({"SEG DENSIDAD": {"nan": np.nan, "None": np.nan, "": np.nan}})
    agg_turn_seg = agg_turn_seg.dropna(subset=["SEG DENSIDAD", "METRIC_VAL"]).copy()

    if agg_turn_seg.empty:
        st.info("No hay datos suficientes para SEG DENSIDAD.")
    else:
        fig_seg = px.box(
            agg_turn_seg,
            x="SEG DENSIDAD",
            y="METRIC_VAL",
            points="outliers",
            title=f"{metric_seg_pick} por SEG DENSIDAD"
        )
        fig_seg.update_layout(
            xaxis=dict(type="category", title="SEG DENSIDAD"),
            yaxis=dict(title=metric_seg_pick)
        )
        st.plotly_chart(fig_seg, use_container_width=True)

        anova_seg = analyze_variance_by_group(agg_turn_seg, "SEG DENSIDAD", "METRIC_VAL")
        render_variance_metrics(anova_seg)
        render_group_descriptive_summary(agg_turn_seg, "SEG DENSIDAD", "METRIC_VAL")

    st.divider()

    st.subheader("Biometría")
    st.caption("Compara SUELO vs MACETA para una variable biométrica.")

    biom_col_pick = st.selectbox(
        "Variable biométrica",
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
            st.info("No hay datos suficientes.")
        else:
            fig_bio = px.box(
                bio_df,
                x="SIEMBRA FINAL",
                y=biom_col_pick,
                points="outliers",
                title=f"{biom_col_pick} por SIEMBRA"
            )
            fig_bio.update_layout(
                xaxis=dict(type="category", title="SIEMBRA"),
                yaxis=dict(title=biom_col_pick)
            )
            st.plotly_chart(fig_bio, use_container_width=True)

            anova_bio = analyze_variance_by_group(bio_df, "SIEMBRA FINAL", biom_col_pick)
            render_variance_metrics(anova_bio)
            render_group_descriptive_summary(bio_df, "SIEMBRA FINAL", biom_col_pick)

    with bio_right:
        bio_summary = build_siembra_final_biometric_summary(dff, biom_col_pick)
        if bio_summary.empty:
            st.info("No hay resumen.")
        else:
            st.dataframe(
                bio_summary.style.format({
                    "PROMEDIO": "{:,.2f}",
                    "MEDIANA": "{:,.2f}",
                    "DESV_STD": "{:,.2f}",
                    "MIN": "{:,.2f}",
                    "MAX": "{:,.2f}",
                    "DELTA": "{:,.2f}",
                }),
                use_container_width=True
            )

st.divider()

# --------------------------
# KG/PLANTA
# --------------------------
st.subheader("KG/PLANTA")

kgp = compute_kg_planta_campaign(dff)
if kgp.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    fig_kp = px.line(
        kgp,
        x="CAMPAÑA",
        y="KG/PLANTA",
        markers=True,
        title="KG/PLANTA por campaña"
    )
    fig_kp.update_layout(
        xaxis=dict(type="category", title="CAMPAÑA"),
        yaxis=dict(title="KG/PLANTA")
    )
    st.plotly_chart(fig_kp, use_container_width=True)

st.divider()

# --------------------------
# VARIEDADES
# --------------------------
st.subheader("Variedades")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    top_n = st.slider("Top N variedades", 5, 25, 10)

    level_var = ["VARIEDAD", "CAMPAÑA"]
    agg_v = aggregate_level(dff, level_var, "KG/HA", mode="ratio_kg_area_turno").rename(columns={"y_val": "KG_HA"})
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
        rows.append({"VARIEDAD": var, "KG_HA": ratio_kg_over_unique_turno_area(g)})
    avg_var = pd.DataFrame(rows).merge(freq, on="VARIEDAD", how="left").fillna({"n": 0})
    avg_var = avg_var.sort_values("n", ascending=False).head(top_n)

    fig_rank = px.bar(
        avg_var.sort_values("KG_HA", ascending=True),
        x="KG_HA", y="VARIEDAD",
        orientation="h",
        title="Ranking KG/HA"
    )
    fig_rank.update_layout(xaxis=dict(title="KG/HA"), yaxis=dict(title="VARIEDAD"))
    st.plotly_chart(fig_rank, use_container_width=True)

    top_vars = avg_var["VARIEDAD"].tolist()
    hm = agg_v[agg_v["VARIEDAD"].isin(top_vars)].copy()
    pivot = hm.pivot_table(index="VARIEDAD", columns="CAMPAÑA", values="KG_HA", aggfunc="mean")
    pivot = pivot.reindex(index=top_vars)

    fig_hm = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=pivot.index.tolist(),
            colorbar=dict(title="KG/HA"),
        )
    )
    fig_hm.update_layout(
        title="VARIEDAD x CAMPAÑA",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(type="category", title="CAMPAÑA"),
        yaxis=dict(title="VARIEDAD")
    )
    st.plotly_chart(fig_hm, use_container_width=True)

st.divider()

# --------------------------
# KG/HA por EDAD
# --------------------------
st.subheader("KG/HA por edad")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    rows = []
    for (camp, edad), g in dff.groupby(["CAMPAÑA", "EDAD PLANTA FINAL"], dropna=False):
        rows.append({
            "CAMPAÑA": str(camp),
            "EDAD PLANTA FINAL": str(edad),
            "KG_HA": ratio_kg_over_unique_turno_area(g)
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
        y="KG_HA",
        color="EDAD PLANTA FINAL",
        barmode="group",
        category_orders={"CAMPAÑA": campaign_order, "EDAD PLANTA FINAL": order_age},
        title="KG/HA por edad y campaña"
    )
    fig_agecamp.update_layout(
        xaxis=dict(type="category", categoryorder="array", categoryarray=campaign_order, title="CAMPAÑA"),
        yaxis=dict(title="KG/HA")
    )
    st.plotly_chart(fig_agecamp, use_container_width=True)

st.divider()

# --------------------------
# NUEVA VISTA: MAX / MIN KG/HA
# --------------------------
st.subheader("MAX / MIN de KG/HA")
st.caption("Identifica el máximo y mínimo de KG/HA según TURNO o CAMPO. Ahora la visualización se muestra por campaña.")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    comp_candidates = get_comparative_candidates(dff)

    kg_left, kg_right = st.columns([0.30, 0.70])

    with kg_left:
        level_mode_kgha = st.selectbox(
            "Filtro de análisis",
            ["TURNO", "CAMPO"],
            key="level_mode_kgha"
        )

        default_comp_kgha = ["BROTES TOTALES"] if "BROTES TOTALES" in comp_candidates else comp_candidates[:1]
        comp_vars_kgha = st.multiselect(
            "Variables comparativas",
            comp_candidates,
            default=default_comp_kgha,
            key="comp_vars_kgha"
        )

    with kg_right:
        if not comp_vars_kgha:
            st.info("Selecciona al menos una variable comparativa.")
        else:
            summary_kgha, detail_kgha = build_entity_maxmin_summary(
                dff,
                level_mode=level_mode_kgha,
                metric_name="KG/HA",
                comp_vars=comp_vars_kgha
            )

            if summary_kgha.empty:
                st.info("No hay datos suficientes para calcular MAX/MIN de KG/HA.")
            else:
                fmt_dict = {
                    "KG/HA": "{:,.4f}",
                    "KG_TOTAL": "{:,.4f}",
                    "HA_TURNO_UNICA": "{:,.4f}",
                }
                for comp_var in comp_vars_kgha:
                    fmt_dict[comp_var] = "{:,.4f}"

                st.dataframe(
                    summary_kgha.style.format(fmt_dict),
                    use_container_width=True
                )
                render_maxmin_chart(summary_kgha, "KG/HA", comp_vars_kgha, level_mode_kgha)

st.divider()

# --------------------------
# NUEVA VISTA: MAX / MIN PESO
# --------------------------
st.subheader("MAX / MIN de PESO")
st.caption("Identifica el máximo y mínimo de PESO según TURNO o CAMPO. Esta vista mantiene la misma lógica completa que la de KG/HA y se muestra por campaña.")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    comp_candidates_peso = get_comparative_candidates(dff)

    pe_left, pe_right = st.columns([0.30, 0.70])

    with pe_left:
        level_mode_peso = st.selectbox(
            "Filtro de análisis ",
            ["TURNO", "CAMPO"],
            key="level_mode_peso"
        )

        default_comp_peso = ["BROTES TOTALES"] if "BROTES TOTALES" in comp_candidates_peso else comp_candidates_peso[:1]
        comp_vars_peso = st.multiselect(
            "Variables comparativas ",
            comp_candidates_peso,
            default=default_comp_peso,
            key="comp_vars_peso"
        )

    with pe_right:
        if not comp_vars_peso:
            st.info("Selecciona al menos una variable comparativa.")
        else:
            summary_peso, detail_peso = build_entity_maxmin_summary(
                dff,
                level_mode=level_mode_peso,
                metric_name="PESO",
                comp_vars=comp_vars_peso
            )

            if summary_peso.empty:
                st.info("No hay datos suficientes para calcular MAX/MIN de PESO.")
            else:
                fmt_dict = {
                    "PESO": "{:,.4f}",
                    "KG_TOTAL": "{:,.4f}",
                    "HA_TURNO_UNICA": "{:,.4f}",
                }
                for comp_var in comp_vars_peso:
                    fmt_dict[comp_var] = "{:,.4f}"

                st.dataframe(
                    summary_peso.style.format(fmt_dict),
                    use_container_width=True
                )
                render_maxmin_chart(summary_peso, "PESO", comp_vars_peso, level_mode_peso)

st.divider()

# --------------------------
# CORRELACIONES
# --------------------------
st.subheader("Correlaciones")
fig_corr = corr_heatmap(dff)
st.plotly_chart(fig_corr, use_container_width=True)
