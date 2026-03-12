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
DATA_FILE = "CONSOLIDADO 2022-2026.xlsb"

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

GENERAL_METRIC_OPTIONS = [
    "KG/HA",
    "KG/PLANTA",
    "PESO BAYA (g)",
]

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
    return pd.read_excel(path, sheet_name=sheet, engine="pyxlsb")

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

def get_general_metric_config(metric_name: str) -> dict:
    metric_name = str(metric_name).strip()

    if metric_name == "KG/HA":
        return {
            "metric_name": "KG/HA",
            "source_col": "KG/HA",
            "agg_mode": "ratio_kg_area_turno",
            "y_title": "KG/HA"
        }

    if metric_name == "KG/PLANTA":
        return {
            "metric_name": "KG/PLANTA",
            "source_col": "KG/PLANTA",
            "agg_mode": "ratio_kg_planta_turno",
            "y_title": "KG/PLANTA"
        }

    if metric_name == "PESO BAYA (g)":
        return {
            "metric_name": "PESO BAYA (g)",
            "source_col": "PESO BAYA (g)",
            "agg_mode": "weighted_kg",
            "y_title": "PESO BAYA (g)"
        }

    return {
        "metric_name": "KG/HA",
        "source_col": "KG/HA",
        "agg_mode": "ratio_kg_area_turno",
        "y_title": "KG/HA"
    }

def compute_general_metric_value(df_subset: pd.DataFrame, metric_name: str) -> float:
    metric_name = str(metric_name).strip()

    if metric_name == "KG/HA":
        return ratio_kg_over_unique_turno_area(df_subset)

    if metric_name == "KG/PLANTA":
        return ratio_kg_planta_over_unique_turno(df_subset)

    if metric_name == "PESO BAYA (g)":
        return weighted_mean(df_subset["PESO BAYA (g)"], df_subset["kilogramos"])

    return np.nan

def build_boxplot_metric_df(dff: pd.DataFrame, group_col: str, metric_name: str) -> pd.DataFrame:
    cfg = get_general_metric_config(metric_name)
    level_cols = UNIT_COLS_BASE + [group_col]

    agg_df = aggregate_level(
        dff,
        level_cols,
        cfg["source_col"],
        mode=cfg["agg_mode"]
    ).rename(columns={"y_val": "METRIC_VAL"})

    agg_df[group_col] = agg_df[group_col].astype(str).str.strip()
    agg_df = agg_df.replace({group_col: {"nan": np.nan, "None": np.nan, "": np.nan}})
    agg_df = agg_df.dropna(subset=[group_col, "METRIC_VAL"]).copy()

    return agg_df

def build_evolution_detail_table(
    evol_df: pd.DataFrame,
    metric_name: str,
    comp_vars: list,
    level_mode: str
) -> pd.DataFrame:
    if evol_df.empty:
        return pd.DataFrame()

    out = evol_df.copy()

    out["CAMPAÑA"] = out["CAMPAÑA"].astype(str)
    out["CODIGO"] = out.apply(lambda r: _build_entity_code(r, level_mode), axis=1)

    keep_cols = ["TIPO_REF", "CAMPAÑA", "FUNDO", "ETAPA", "CAMPO"]
    if level_mode == "TURNO":
        keep_cols.append("TURNO")
    keep_cols += ["VARIEDAD", "CODIGO", "METRICA", "KG_TOTAL", "AREA_TURNO_UNICA"]

    for c in comp_vars:
        if c in out.columns:
            keep_cols.append(c)

    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].copy()

    out = out.rename(columns={
        "TIPO_REF": "TIPO",
        "METRICA": metric_name,
        "AREA_TURNO_UNICA": "HA_TURNO_UNICA"
    })

    campaign_order = _sort_campaign_categories(out["CAMPAÑA"])
    out["CAMPAÑA"] = pd.Categorical(out["CAMPAÑA"], categories=campaign_order, ordered=True)
    out = out.sort_values(["TIPO", "CAMPAÑA"]).reset_index(drop=True)

    return out

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

        metric_val = compute_general_metric_value(g, metric_name)

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
        return pd.DataFrame(), detail

    detail = detail.dropna(subset=["METRICA"]).copy()
    if detail.empty:
        return pd.DataFrame(), detail

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

def _build_entity_code(row: pd.Series, level_mode: str) -> str:
    etapa = str(row.get("ETAPA", "")).strip()
    campo = str(row.get("CAMPO", "")).strip()

    if level_mode == "TURNO":
        turno = str(row.get("TURNO", "")).strip()
        return f"{etapa},{campo},{turno}"
    return f"{etapa},{campo}"

def _build_entity_key(row: pd.Series, level_mode: str) -> str:
    if level_mode == "TURNO":
        parts = [
            str(row.get("FUNDO", "")).strip(),
            str(row.get("ETAPA", "")).strip(),
            str(row.get("CAMPO", "")).strip(),
            str(row.get("TURNO", "")).strip(),
            str(row.get("VARIEDAD", "")).strip(),
        ]
    else:
        parts = [
            str(row.get("FUNDO", "")).strip(),
            str(row.get("ETAPA", "")).strip(),
            str(row.get("CAMPO", "")).strip(),
            str(row.get("VARIEDAD", "")).strip(),
        ]
    return "||".join(parts)

def _parse_entity_key(entity_key: str, level_mode: str) -> dict:
    parts = str(entity_key).split("||")
    if level_mode == "TURNO":
        parts = (parts + [""] * 5)[:5]
        return {
            "FUNDO": parts[0],
            "ETAPA": parts[1],
            "CAMPO": parts[2],
            "TURNO": parts[3],
            "VARIEDAD": parts[4],
        }
    parts = (parts + [""] * 4)[:4]
    return {
        "FUNDO": parts[0],
        "ETAPA": parts[1],
        "CAMPO": parts[2],
        "VARIEDAD": parts[3],
    }

def render_maxmin_chart(summary_df: pd.DataFrame, metric_name: str, comp_vars: list, level_mode: str):
    if summary_df.empty:
        st.info("No hay datos suficientes para mostrar la comparación MAX/MIN.")
        return

    df_plot = summary_df.copy()
    df_plot["CAMPAÑA"] = df_plot["CAMPAÑA"].astype(str)

    campaign_order = _sort_campaign_categories(df_plot["CAMPAÑA"])
    fig = go.Figure()

    color_map = {"MAX": "#156cc2", "MIN": "#8bbcf0"}

    camp_pos = {camp: i * 2.0 for i, camp in enumerate(campaign_order)}
    offset_map = {"MAX": -0.35, "MIN": 0.35}
    bar_width = 0.42

    for tipo in ["MAX", "MIN"]:
        d = df_plot[df_plot["TIPO"] == tipo].copy()
        d = d.set_index("CAMPAÑA").reindex(campaign_order).reset_index()

        x_vals = []
        y_vals = []
        text_vals = []
        customdata = []

        for _, r in d.iterrows():
            metric_val = r.get(metric_name, np.nan)
            if pd.isna(metric_val):
                continue

            camp = str(r["CAMPAÑA"])
            x_pos = camp_pos[camp] + offset_map[tipo]
            entity_code = _build_entity_code(r, level_mode)

            x_vals.append(x_pos)
            y_vals.append(metric_val)
            text_vals.append(f"{entity_code}<br>{metric_val:,.0f}")
            customdata.append([camp, tipo, entity_code])

        fig.add_trace(go.Bar(
            x=x_vals,
            y=y_vals,
            name=f"{metric_name} {tipo}",
            yaxis="y1",
            width=bar_width,
            marker_color=color_map[tipo],
            text=text_vals,
            textposition="outside",
            textfont=dict(size=11),
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "Campaña: %{customdata[0]}<br>"
                "Tipo: %{customdata[1]}<br>"
                f"Código: " + "%{customdata[2]}<br>"
                f"{metric_name}: %{{y:,.2f}}<extra></extra>"
            )
        ))

    for comp_var in comp_vars:
        x_vals = []
        y_vals = []
        text_vals = []
        customdata = []

        for camp in campaign_order:
            pts = []

            row_max = df_plot[(df_plot["CAMPAÑA"] == str(camp)) & (df_plot["TIPO"] == "MAX")]
            if not row_max.empty:
                row_max = row_max.iloc[0]
                val_max = row_max.get(comp_var, np.nan)
                if pd.notna(val_max):
                    pts.append({
                        "x": camp_pos[str(camp)] + offset_map["MAX"],
                        "y": val_max,
                        "text": f"{val_max:,.0f}",
                        "camp": str(camp),
                        "tipo": "MAX",
                        "code": _build_entity_code(row_max, level_mode)
                    })

            row_min = df_plot[(df_plot["CAMPAÑA"] == str(camp)) & (df_plot["TIPO"] == "MIN")]
            if not row_min.empty:
                row_min = row_min.iloc[0]
                val_min = row_min.get(comp_var, np.nan)
                if pd.notna(val_min):
                    pts.append({
                        "x": camp_pos[str(camp)] + offset_map["MIN"],
                        "y": val_min,
                        "text": f"{val_min:,.0f}",
                        "camp": str(camp),
                        "tipo": "MIN",
                        "code": _build_entity_code(row_min, level_mode)
                    })

            if len(pts) == 0:
                continue

            if len(pts) == 1:
                p = pts[0]
                x_vals.extend([p["x"], None])
                y_vals.extend([p["y"], None])
                text_vals.extend([p["text"], None])
                customdata.extend([
                    [p["camp"], p["tipo"], p["code"]],
                    [None, None, None]
                ])
            else:
                pts = sorted(pts, key=lambda z: z["x"])
                for p in pts:
                    x_vals.append(p["x"])
                    y_vals.append(p["y"])
                    text_vals.append(p["text"])
                    customdata.append([p["camp"], p["tipo"], p["code"]])
                x_vals.append(None)
                y_vals.append(None)
                text_vals.append(None)
                customdata.append([None, None, None])

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers+text",
            name=comp_var,
            yaxis="y2",
            text=text_vals,
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(size=8),
            customdata=customdata,
            hovertemplate=(
                "Campaña: %{customdata[0]}<br>"
                "Tipo: %{customdata[1]}<br>"
                f"Código: " + "%{customdata[2]}<br>"
                f"{comp_var}: %{{y:,.2f}}<extra></extra>"
            ),
            connectgaps=False
        ))

    tickvals = [camp_pos[c] for c in campaign_order]
    ticktext = campaign_order

    fig.update_layout(
        title=f"MAX vs MIN de {metric_name} por {level_mode} + {', '.join(comp_vars)}",
        xaxis=dict(
            title="CAMPAÑA",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[min(tickvals) - 0.9, max(tickvals) + 0.9] if tickvals else None
        ),
        yaxis=dict(title=metric_name),
        yaxis2=dict(title="Variables comparativas", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        height=620,
        margin=dict(t=95, b=70, l=40, r=40),
        bargap=0.30
    )
    st.plotly_chart(fig, use_container_width=True)

def build_evolution_maxmin_summary(
    df_input: pd.DataFrame,
    level_mode: str,
    metric_name: str,
    comp_vars: list,
    campaign_anchor: str
):
    if df_input.empty or not campaign_anchor:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary_base, detail_all = build_entity_maxmin_summary(
        df_input=df_input,
        level_mode=level_mode,
        metric_name=metric_name,
        comp_vars=comp_vars
    )

    if summary_base.empty or detail_all.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    campaign_anchor = str(campaign_anchor).strip()
    base_rows = summary_base[summary_base["CAMPAÑA"].astype(str) == campaign_anchor].copy()

    if base_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    base_rows["ENTITY_KEY"] = base_rows.apply(lambda r: _build_entity_key(r, level_mode), axis=1)

    detail = detail_all.copy()
    detail["CAMPAÑA"] = detail["CAMPAÑA"].astype(str)
    detail["ENTITY_KEY"] = detail.apply(lambda r: _build_entity_key(r, level_mode), axis=1)

    selected_keys = base_rows["ENTITY_KEY"].dropna().unique().tolist()
    evol = detail[detail["ENTITY_KEY"].isin(selected_keys)].copy()

    if evol.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    key_to_tipo = base_rows.set_index("ENTITY_KEY")["TIPO"].to_dict()
    evol["TIPO_REF"] = evol["ENTITY_KEY"].map(key_to_tipo)

    campaign_order = _sort_campaign_categories(evol["CAMPAÑA"])
    evol["CAMPAÑA"] = pd.Categorical(evol["CAMPAÑA"], categories=campaign_order, ordered=True)
    evol = evol.sort_values(["TIPO_REF", "CAMPAÑA"]).reset_index(drop=True)

    base_rows["CAMPAÑA"] = pd.Categorical(base_rows["CAMPAÑA"].astype(str), categories=_sort_campaign_categories(base_rows["CAMPAÑA"].astype(str)), ordered=True)
    base_rows = base_rows.sort_values(["CAMPAÑA", "TIPO"]).reset_index(drop=True)

    return base_rows, evol, detail_all

def render_evolution_chart(base_df: pd.DataFrame, evol_df: pd.DataFrame, metric_name: str, comp_vars: list, level_mode: str, campaign_anchor: str):
    if base_df.empty or evol_df.empty:
        st.info("No hay datos suficientes para mostrar la evolución.")
        return

    df_plot = evol_df.copy()
    df_plot["CAMPAÑA"] = df_plot["CAMPAÑA"].astype(str)
    campaign_order = _sort_campaign_categories(df_plot["CAMPAÑA"])

    fig = go.Figure()

    color_metric_map = {"MAX": "#156cc2", "MIN": "#8bbcf0"}

    for tipo_ref in ["MAX", "MIN"]:
        d = df_plot[df_plot["TIPO_REF"] == tipo_ref].copy()
        d = d.sort_values("CAMPAÑA")

        x_vals = []
        y_vals = []
        text_vals = []
        customdata = []

        for _, r in d.iterrows():
            metric_val = r.get("METRICA", np.nan)
            if pd.isna(metric_val):
                continue

            entity_code = _build_entity_code(r, level_mode)
            x_vals.append(str(r["CAMPAÑA"]))
            y_vals.append(metric_val)
            text_vals.append(f"{entity_code}<br>{metric_val:,.0f}")
            customdata.append([str(r["CAMPAÑA"]), tipo_ref, entity_code])

        fig.add_trace(go.Bar(
            x=x_vals,
            y=y_vals,
            name=f"{metric_name} {tipo_ref}",
            yaxis="y1",
            marker_color=color_metric_map[tipo_ref],
            text=text_vals,
            textposition="outside",
            textfont=dict(size=11),
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "Campaña: %{customdata[0]}<br>"
                "Referencia: %{customdata[1]}<br>"
                f"Código: " + "%{customdata[2]}<br>"
                f"{metric_name}: %{{y:,.2f}}<extra></extra>"
            )
        ))

    dash_map = {"MAX": "solid", "MIN": "dot"}

    for comp_var in comp_vars:
        for tipo_ref in ["MAX", "MIN"]:
            d = df_plot[df_plot["TIPO_REF"] == tipo_ref].copy()
            d = d.sort_values("CAMPAÑA")

            x_vals = []
            y_vals = []
            text_vals = []
            customdata = []

            for _, r in d.iterrows():
                comp_val = r.get(comp_var, np.nan)
                if pd.isna(comp_val):
                    continue

                entity_code = _build_entity_code(r, level_mode)
                x_vals.append(str(r["CAMPAÑA"]))
                y_vals.append(comp_val)
                text_vals.append(f"{comp_val:,.0f}")
                customdata.append([str(r["CAMPAÑA"]), tipo_ref, entity_code])

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers+text",
                name=f"{comp_var} {tipo_ref}",
                yaxis="y2",
                text=text_vals,
                textposition="top center",
                textfont=dict(size=10),
                marker=dict(size=8),
                line=dict(dash=dash_map[tipo_ref]),
                customdata=customdata,
                hovertemplate=(
                    "Campaña: %{customdata[0]}<br>"
                    "Referencia: %{customdata[1]}<br>"
                    f"Código: " + "%{customdata[2]}<br>"
                    f"{comp_var}: %{{y:,.2f}}<extra></extra>"
                ),
                connectgaps=False
            ))

    fig.update_layout(
        title=f"Evolución de MAX/MIN de {metric_name} tomando como referencia la campaña {campaign_anchor}",
        xaxis=dict(
            title="CAMPAÑA",
            type="category",
            categoryorder="array",
            categoryarray=campaign_order
        ),
        yaxis=dict(title=metric_name),
        yaxis2=dict(title="Variables comparativas", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        height=650,
        margin=dict(t=95, b=70, l=40, r=40),
        bargap=0.25,
        barmode="group"
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

    metric_general_pick = st.selectbox(
        "MÉTRICA",
        GENERAL_METRIC_OPTIONS,
        index=0
    )

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
    metric_cfg_box = get_general_metric_config(metric_general_pick)
    y_metric_label = metric_cfg_box["y_title"]

    st.caption(f"Los tres boxplots responden a la métrica general seleccionada: {y_metric_label}")

    b1, b2 = st.columns(2)

    with b1:
        agg_turn_siem = build_boxplot_metric_df(dff, "SIEMBRA FINAL", metric_general_pick)

        if agg_turn_siem.empty:
            st.info("No hay datos suficientes para SIEMBRA FINAL.")
        else:
            fig_siem = px.box(
                agg_turn_siem,
                x="SIEMBRA FINAL",
                y="METRIC_VAL",
                points="outliers",
                title=f"{y_metric_label} por SIEMBRA"
            )
            fig_siem.update_layout(
                xaxis=dict(type="category", title="SIEMBRA"),
                yaxis=dict(title=y_metric_label)
            )
            st.plotly_chart(fig_siem, use_container_width=True)

            anova_siem = analyze_variance_by_group(agg_turn_siem, "SIEMBRA FINAL", "METRIC_VAL")
            render_variance_metrics(anova_siem)
            render_group_descriptive_summary(agg_turn_siem, "SIEMBRA FINAL", "METRIC_VAL")

    with b2:
        agg_turn_age = build_boxplot_metric_df(dff, "EDAD PLANTA FINAL", metric_general_pick)

        if agg_turn_age.empty:
            st.info("No hay datos suficientes para EDAD PLANTA FINAL.")
        else:
            order_age = ["1", "2", "3+"]
            fig_age = px.box(
                agg_turn_age,
                x="EDAD PLANTA FINAL",
                y="METRIC_VAL",
                category_orders={"EDAD PLANTA FINAL": order_age},
                points="outliers",
                title=f"{y_metric_label} por EDAD"
            )
            fig_age.update_layout(
                xaxis=dict(type="category", title="EDAD"),
                yaxis=dict(title=y_metric_label)
            )
            st.plotly_chart(fig_age, use_container_width=True)

            anova_age = analyze_variance_by_group(agg_turn_age, "EDAD PLANTA FINAL", "METRIC_VAL")
            render_variance_metrics(anova_age)
            render_group_descriptive_summary(agg_turn_age, "EDAD PLANTA FINAL", "METRIC_VAL")

    st.divider()

    st.subheader("SEG DENSIDAD")
    st.caption(f"Compara SEG DENSIDAD usando la métrica general seleccionada: {y_metric_label}")

    agg_turn_seg = build_boxplot_metric_df(dff, "SEG DENSIDAD", metric_general_pick)

    if agg_turn_seg.empty:
        st.info("No hay datos suficientes para SEG DENSIDAD.")
    else:
        fig_seg = px.box(
            agg_turn_seg,
            x="SEG DENSIDAD",
            y="METRIC_VAL",
            points="outliers",
            title=f"{y_metric_label} por SEG DENSIDAD"
        )
        fig_seg.update_layout(
            xaxis=dict(type="category", title="SEG DENSIDAD"),
            yaxis=dict(title=y_metric_label)
        )
        st.plotly_chart(fig_seg, use_container_width=True)

        anova_seg = analyze_variance_by_group(agg_turn_seg, "SEG DENSIDAD", "METRIC_VAL")
        render_variance_metrics(anova_seg)
        render_group_descriptive_summary(agg_turn_seg, "SEG DENSIDAD", "METRIC_VAL")

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
# VISTA DINÁMICA: MAX / MIN
# --------------------------
st.subheader(f"MAX / MIN de {metric_general_pick}")
st.caption("Identifica el máximo y mínimo de la métrica seleccionada según TURNO o CAMPO.")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    comp_candidates_dyn = get_comparative_candidates(dff)
    campaign_options_mm_dyn = ["TODAS"] + _sort_campaign_categories(dff["CAMPAÑA"].astype(str))

    mm_left, mm_right = st.columns([0.30, 0.70])

    with mm_left:
        level_mode_dyn = st.selectbox(
            "Filtro de análisis",
            ["TURNO", "CAMPO"],
            key="level_mode_dyn"
        )

        campaign_filter_dyn = st.selectbox(
            "CAMPAÑA",
            campaign_options_mm_dyn,
            index=0,
            key="campaign_filter_dyn"
        )

        default_comp_dyn = ["BROTES TOTALES"] if "BROTES TOTALES" in comp_candidates_dyn else comp_candidates_dyn[:1]
        comp_vars_dyn = st.multiselect(
            "Variables comparativas",
            comp_candidates_dyn,
            default=default_comp_dyn,
            key="comp_vars_dyn"
        )

    with mm_right:
        if not comp_vars_dyn:
            st.info("Selecciona al menos una variable comparativa.")
        else:
            summary_dyn, detail_dyn = build_entity_maxmin_summary(
                dff,
                level_mode=level_mode_dyn,
                metric_name=metric_general_pick,
                comp_vars=comp_vars_dyn
            )

            if campaign_filter_dyn != "TODAS" and not summary_dyn.empty:
                summary_dyn = summary_dyn[summary_dyn["CAMPAÑA"].astype(str) == str(campaign_filter_dyn)].copy()

            if summary_dyn.empty:
                st.info(f"No hay datos suficientes para calcular MAX/MIN de {metric_general_pick}.")
            else:
                fmt_dict = {
                    metric_general_pick: "{:,.4f}",
                    "KG_TOTAL": "{:,.4f}",
                    "HA_TURNO_UNICA": "{:,.4f}",
                }
                for comp_var in comp_vars_dyn:
                    fmt_dict[comp_var] = "{:,.4f}"

                st.dataframe(
                    summary_dyn.style.format(fmt_dict),
                    use_container_width=True
                )
                render_maxmin_chart(summary_dyn, metric_general_pick, comp_vars_dyn, level_mode_dyn)

st.divider()

# --------------------------
# VISTA DINÁMICA: EVOLUCIÓN DE MAX/MIN
# --------------------------
st.subheader(f"EVOLUCIÓN DE MAX/MIN DE {metric_general_pick}")
st.caption("Toma una campaña como referencia, identifica su MAX y MIN, y sigue esos mismos elementos hacia atrás y hacia adelante en las demás campañas.")

if dff.empty:
    st.warning("No hay datos con los filtros actuales.")
else:
    comp_candidates_ev_dyn = get_comparative_candidates(dff)
    campaign_options_ev_dyn = _sort_campaign_categories(dff["CAMPAÑA"].astype(str))

    ev_left, ev_right = st.columns([0.30, 0.70])

    with ev_left:
        level_mode_ev_dyn = st.selectbox(
            "Filtro de análisis ",
            ["TURNO", "CAMPO"],
            key="level_mode_ev_dyn"
        )

        default_comp_ev_dyn = ["BROTES TOTALES"] if "BROTES TOTALES" in comp_candidates_ev_dyn else comp_candidates_ev_dyn[:1]
        comp_vars_ev_dyn = st.multiselect(
            "Variables comparativas ",
            comp_candidates_ev_dyn,
            default=default_comp_ev_dyn,
            key="comp_vars_ev_dyn"
        )

        campaign_anchor_dyn = st.selectbox(
            "CAMPAÑA:",
            campaign_options_ev_dyn,
            index=len(campaign_options_ev_dyn) - 1 if campaign_options_ev_dyn else 0,
            key="campaign_anchor_dyn"
        )

    with ev_right:
        if not comp_vars_ev_dyn:
            st.info("Selecciona al menos una variable comparativa.")
        elif not campaign_anchor_dyn:
            st.info("Selecciona una campaña base.")
        else:
            summary_base_dyn, evol_dyn, detail_all_dyn = build_evolution_maxmin_summary(
                dff,
                level_mode=level_mode_ev_dyn,
                metric_name=metric_general_pick,
                comp_vars=comp_vars_ev_dyn,
                campaign_anchor=campaign_anchor_dyn
            )

            if summary_base_dyn.empty or evol_dyn.empty:
                st.info(f"No hay datos suficientes para calcular la evolución de MAX/MIN de {metric_general_pick}.")
            else:
                fmt_dict_base = {
                    metric_general_pick: "{:,.4f}",
                    "KG_TOTAL": "{:,.4f}",
                    "HA_TURNO_UNICA": "{:,.4f}",
                }
                for comp_var in comp_vars_ev_dyn:
                    fmt_dict_base[comp_var] = "{:,.4f}"

                st.markdown(f"**Base seleccionada en campaña {campaign_anchor_dyn}**")
                st.dataframe(
                    summary_base_dyn.style.format(fmt_dict_base),
                    use_container_width=True
                )

                st.markdown("**Evolución completa en tabla**")
                evol_table = build_evolution_detail_table(
                    evol_df=evol_dyn,
                    metric_name=metric_general_pick,
                    comp_vars=comp_vars_ev_dyn,
                    level_mode=level_mode_ev_dyn
                )

                if not evol_table.empty:
                    fmt_dict_evol = {
                        metric_general_pick: "{:,.4f}",
                        "KG_TOTAL": "{:,.4f}",
                        "HA_TURNO_UNICA": "{:,.4f}",
                    }
                    for comp_var in comp_vars_ev_dyn:
                        if comp_var in evol_table.columns:
                            fmt_dict_evol[comp_var] = "{:,.4f}"

                    st.dataframe(
                        evol_table.style.format(fmt_dict_evol),
                        use_container_width=True
                    )

                render_evolution_chart(
                    summary_base_dyn,
                    evol_dyn,
                    metric_general_pick,
                    comp_vars_ev_dyn,
                    level_mode_ev_dyn,
                    campaign_anchor_dyn
                )

st.divider()

# --------------------------
# CORRELACIONES
# --------------------------
st.subheader("Correlaciones")
fig_corr = corr_heatmap(dff)
st.plotly_chart(fig_corr, use_container_width=True)
