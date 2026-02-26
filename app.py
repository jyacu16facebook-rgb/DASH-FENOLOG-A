# app.py
# ============================================================
# DASHBOARD: Drivers fenológicos/estructura asociados a KG/HA
# Fuente: Excel subido por el usuario | Hoja: "DATA"
# ============================================================

import re
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


# -------------------------
# CONFIG STREAMLIT
# -------------------------
st.set_page_config(page_title="Fenología vs Rendimiento (KG/HA)", layout="wide")
st.title("🫐 Fenología y estructura vs rendimiento (KG/HA) | Campañas 2022–2025")


# -------------------------
# PARAMETROS / DICCIONARIOS
# -------------------------
HOJA = "DATA"

COLUMNAS_ESPERADAS = [
    "AÑO", "CAMPAÑA", "SEMANA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
    "kilogramos", "FLORES", "FRUTO CUAJADO", "FRUTO VERDE", "Ha COSECHADA", "Ha TURNO",
    "KG/HA", "DENSIDAD", "FRUTO MADURO", "FRUTO ROSADO", "FRUTO CREMOSO",
    "PESO BAYA (g)", "PESO BAYA CREMOSO (g)", "CALIBRE BAYA (mm)", "CALIBRE CREMOSO (mm)",
    "SEMANA DE SIEMBRA", "FECHA FIN DE SIEMBRA", "TIPO PODA", "FECHA PODA",
    "MADERAS PRINCIPALES", "CARGADORES", "N° RAMAS VEGETATIVAS", "RAMAS TOTALES", "TERMINALES",
    "EDAD PLANTA", "EDAD PLANTA FINAL",
    "BP_N_BROTES_ULT", "BP_LONG_B1_ULT", "BP_LONG_B2_ULT", "BP_DIAM_B1_ULT", "BP_DIAM_B2_ULT",
    "BS_N_BROTES_ULT", "BS_LONG_B1_ULT", "BS_LONG_B2_ULT", "BS_DIAM_B1_ULT", "BS_DIAM_B2_ULT",
    "BT_N_BROTES_ULT", "BT_LONG_B1_ULT", "BT_LONG_B2_ULT", "BT_DIAM_B1_ULT", "BT_DIAM_B2_ULT",
    "ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT", "SIEMBRA"
]

CATEGORICAS_BASE = [
    "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD", "TIPO PODA", "SIEMBRA", "EDAD PLANTA FINAL"
]

FENOLOGIA_CORE = [
    "FLORES", "FRUTO CUAJADO", "FRUTO VERDE", "FRUTO CREMOSO", "FRUTO ROSADO", "FRUTO MADURO",
    "PESO BAYA (g)", "CALIBRE BAYA (mm)", "PESO BAYA CREMOSO (g)", "CALIBRE CREMOSO (mm)"
]

ESTRUCTURA_CORE = [
    "MADERAS PRINCIPALES", "CARGADORES", "RAMAS TOTALES", "TERMINALES",
    "ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT"
]

BROTES_CORE = [
    "BP_N_BROTES_ULT", "BP_LONG_B1_ULT", "BP_LONG_B2_ULT", "BP_DIAM_B1_ULT", "BP_DIAM_B2_ULT",
    "BS_N_BROTES_ULT", "BS_LONG_B1_ULT", "BS_LONG_B2_ULT", "BS_DIAM_B1_ULT", "BS_DIAM_B2_ULT",
    "BT_N_BROTES_ULT", "BT_LONG_B1_ULT", "BT_LONG_B2_ULT", "BT_DIAM_B1_ULT", "BT_DIAM_B2_ULT",
]


# -------------------------
# UTILIDADES
# -------------------------
def _to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalize_edad_final(x):
    if pd.isna(x):
        return np.nan
    x_str = str(x).strip()
    # normaliza "3" / "3+" / "3 +"
    if re.match(r"^3\s*\+?$", x_str) or "3+" in x_str:
        return "3+"
    if x_str in ["1", "1.0", "01"]:
        return "1"
    if x_str in ["2", "2.0", "02"]:
        return "2"
    return x_str


def _smart_nan_for_zeros(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Heurística: si una columna es casi todo 0 y casi no hay positivos,
    interpretamos 0 como 'no evaluado' y lo pasamos a NaN.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if not pd.api.types.is_numeric_dtype(out[c]):
            continue

        s = out[c]
        nn = s.notna().sum()
        if nn == 0:
            continue

        zeros = (s == 0).sum()
        pos = (s > 0).sum()

        if zeros / nn > 0.85 and pos / nn < 0.10:
            out.loc[out[c] == 0, c] = np.nan
    return out


def wavg(series: pd.Series, weights: pd.Series) -> float:
    """Promedio ponderado robusto."""
    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    m = s.notna() & w.notna() & (w > 0)
    if m.sum() == 0:
        return np.nan
    return float((s[m] * w[m]).sum() / w[m].sum())


@st.cache_data(show_spinner=False)
def cargar_excel(uploaded_file, hoja: str) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name=hoja)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validar_columnas(df: pd.DataFrame):
    faltantes = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
    if faltantes:
        st.warning("⚠️ Columnas esperadas NO encontradas (revisa nombres exactos en tu Excel):")
        st.write(faltantes)


def preparar_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # enteros (semana/año/campaña)
    for c in ["AÑO", "CAMPAÑA", "SEMANA"]:
        if c in d.columns:
            d[c] = _to_numeric_safe(d[c]).astype("Int64")

    # strings categóricas
    for c in CATEGORICAS_BASE:
        if c in d.columns:
            d[c] = d[c].astype("string").str.strip()

    if "EDAD PLANTA FINAL" in d.columns:
        d["EDAD PLANTA FINAL"] = d["EDAD PLANTA FINAL"].apply(_normalize_edad_final).astype("string")

    # fechas
    for c in ["FECHA PODA", "FECHA FIN DE SIEMBRA"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")

    # numéricas
    llaves = ["AÑO", "CAMPAÑA", "SEMANA"] + CATEGORICAS_BASE + ["EDAD PLANTA"]
    num_cols = [c for c in d.columns if c not in llaves and c not in ["FECHA PODA", "FECHA FIN DE SIEMBRA"]]

    for c in num_cols:
        d[c] = _to_numeric_safe(d[c])

    # KG/HA si faltara, intentamos calcular
    if "KG/HA" not in d.columns:
        d["KG/HA"] = np.nan

    if "kilogramos" in d.columns and "Ha TURNO" in d.columns:
        mask_nan = d["KG/HA"].isna()
        denom = d["Ha TURNO"].replace({0: np.nan})
        d.loc[mask_nan, "KG/HA"] = d.loc[mask_nan, "kilogramos"] / denom

    # KG/PLANTA
    if "DENSIDAD" in d.columns and "Ha TURNO" in d.columns and "kilogramos" in d.columns:
        plantas = d["Ha TURNO"].replace({0: np.nan}) * d["DENSIDAD"].replace({0: np.nan})
        d["KG/PLANTA"] = d["kilogramos"] / plantas
    else:
        d["KG/PLANTA"] = np.nan

    # limpiar 0s sospechosos
    candidatos_zeros_missing = (
        ESTRUCTURA_CORE + BROTES_CORE + [
            "PESO BAYA (g)", "PESO BAYA CREMOSO (g)",
            "CALIBRE BAYA (mm)", "CALIBRE CREMOSO (mm)",
        ]
    )
    d = _smart_nan_for_zeros(d, candidatos_zeros_missing)

    return d


def filtrar_df(d: pd.DataFrame, filtros: dict) -> pd.DataFrame:
    out = d.copy()
    for col, val in filtros.items():
        if val is None or col not in out.columns:
            continue
        if isinstance(val, tuple) and len(val) == 2:
            lo, hi = val
            out = out[(out[col].notna()) & (out[col] >= lo) & (out[col] <= hi)]
        elif isinstance(val, list):
            if len(val) > 0:
                out = out[out[col].isin(val)]
        else:
            out = out[out[col] == val]
    return out


def agg_campaign_table(df_f: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla por campaña:
      - KG = suma kilogramos
      - KG/HA ponderado por Ha COSECHADA
      - PESO, CALIBRE ponderados por Ha COSECHADA
    """
    out_rows = []
    for camp, g in df_f.groupby("CAMPAÑA", dropna=False):
        kg_sum = float(pd.to_numeric(g.get("kilogramos"), errors="coerce").sum())
        ha_w = g.get("Ha COSECHADA")
        kg_ha_w = wavg(g.get("KG/HA"), ha_w)
        peso_w = wavg(g.get("PESO BAYA (g)"), ha_w)
        calibre_w = wavg(g.get("CALIBRE BAYA (mm)"), ha_w)

        out_rows.append({
            "CAMPAÑA": int(camp) if pd.notna(camp) else camp,
            "KG": kg_sum,
            "KG/HA (pond Ha COSECHADA)": kg_ha_w,
            "PESO (g) (pond Ha COSECHADA)": peso_w,
            "CALIBRE (mm) (pond Ha COSECHADA)": calibre_w,
        })

    out = pd.DataFrame(out_rows).sort_values("CAMPAÑA")
    return out


def agg_weekly(df_f: pd.DataFrame) -> pd.DataFrame:
    """
    Serie semanal por campaña:
    KG/HA ponderado por Ha COSECHADA
    """
    rows = []
    for (camp, sem), g in df_f.groupby(["CAMPAÑA", "SEMANA"], dropna=False):
        kg_ha_w = wavg(g["KG/HA"], g["Ha COSECHADA"])
        rows.append({"CAMPAÑA": camp, "SEMANA": sem, "KG/HA_pond": kg_ha_w})
    return pd.DataFrame(rows).dropna(subset=["SEMANA"]).sort_values(["CAMPAÑA", "SEMANA"])


def agg_cat_weighted(df_f: pd.DataFrame, cat_col: str, value_col: str = "KG/HA") -> pd.DataFrame:
    """
    Promedio ponderado (por Ha COSECHADA) por categoría.
    """
    rows = []
    for cat, g in df_f.groupby(cat_col, dropna=False):
        rows.append({
            cat_col: cat,
            f"{value_col}_pond": wavg(g[value_col], g["Ha COSECHADA"]),
            "Ha COSECHADA (sum)": float(pd.to_numeric(g["Ha COSECHADA"], errors="coerce").fillna(0).sum()),
            "n": int(len(g))
        })
    out = pd.DataFrame(rows).sort_values(f"{value_col}_pond", ascending=False)
    return out


def agg_variety_rank(df_f: pd.DataFrame, topN: int) -> pd.DataFrame:
    rows = []
    for var, g in df_f.groupby("VARIEDAD", dropna=False):
        rows.append({
            "VARIEDAD": var,
            "KG/HA_pond": wavg(g["KG/HA"], g["Ha COSECHADA"]),
            "Ha COSECHADA (sum)": float(pd.to_numeric(g["Ha COSECHADA"], errors="coerce").fillna(0).sum()),
            "n": int(len(g))
        })
    out = pd.DataFrame(rows).dropna(subset=["VARIEDAD"])
    out = out.sort_values(["KG/HA_pond", "Ha COSECHADA (sum)"], ascending=False).head(topN)
    return out


def agg_variety_vs_campaign(df_f: pd.DataFrame, topN_vars: int = 15) -> pd.DataFrame:
    """
    Heatmap Variedad x Campaña (KG/HA ponderado).
    Para no saturar, se limita a topN_vars por frecuencia.
    """
    top_vars = df_f["VARIEDAD"].value_counts().head(topN_vars).index.tolist()
    tmp = df_f[df_f["VARIEDAD"].isin(top_vars)].copy()

    rows = []
    for (var, camp), g in tmp.groupby(["VARIEDAD", "CAMPAÑA"], dropna=False):
        rows.append({
            "VARIEDAD": var,
            "CAMPAÑA": camp,
            "KG/HA_pond": wavg(g["KG/HA"], g["Ha COSECHADA"])
        })
    return pd.DataFrame(rows)


def turnos_max_min_por_var_camp(df_f: pd.DataFrame, variedad: str):
    """
    Para una VARIEDAD:
    por cada CAMPAÑA -> encuentra TURNO MAX y TURNO MIN según KG/HA ponderado por Ha COSECHADA
    y devuelve tabla + datos para graficar (best vs worst).
    """
    tmp = df_f[df_f["VARIEDAD"] == variedad].copy()
    if tmp.empty:
        return None, None

    # agregamos a nivel CAMPAÑA+VARIEDAD+TURNO
    g_rows = []
    group_cols = ["CAMPAÑA", "VARIEDAD", "TURNO"]
    need_cols = ["KG/HA", "Ha COSECHADA", "MADERAS PRINCIPALES", "CARGADORES", "RAMAS TOTALES"]

    for (camp, var, turno), g in tmp.groupby(group_cols, dropna=False):
        ha_w = g["Ha COSECHADA"]
        g_rows.append({
            "CAMPAÑA": camp,
            "VARIEDAD": var,
            "TURNO": turno,
            "KG/HA_pond": wavg(g["KG/HA"], ha_w),
            "MADERAS PRINCIPALES": wavg(g["MADERAS PRINCIPALES"], ha_w),
            "CARGADORES": wavg(g["CARGADORES"], ha_w),
            "RAMAS TOTALES": wavg(g["RAMAS TOTALES"], ha_w),
            "Ha COSECHADA (sum)": float(pd.to_numeric(ha_w, errors="coerce").fillna(0).sum()),
            "n": int(len(g))
        })

    agg = pd.DataFrame(g_rows).dropna(subset=["KG/HA_pond"])
    if agg.empty:
        return None, None

    out_rows = []
    plot_rows = []
    for camp, gc in agg.groupby("CAMPAÑA", dropna=False):
        gc2 = gc.dropna(subset=["KG/HA_pond"])
        if gc2.empty:
            continue
        best = gc2.loc[gc2["KG/HA_pond"].idxmax()]
        worst = gc2.loc[gc2["KG/HA_pond"].idxmin()]

        out_rows.append({
            "CAMPAÑA": camp,
            "TURNO_MAX": best["TURNO"],
            "KG/HA_MAX (pond)": best["KG/HA_pond"],
            "MADERAS_MAX": best["MADERAS PRINCIPALES"],
            "CARGADORES_MAX": best["CARGADORES"],
            "RAMAS_TOTALES_MAX": best["RAMAS TOTALES"],
            "TURNO_MIN": worst["TURNO"],
            "KG/HA_MIN (pond)": worst["KG/HA_pond"],
            "MADERAS_MIN": worst["MADERAS PRINCIPALES"],
            "CARGADORES_MIN": worst["CARGADORES"],
            "RAMAS_TOTALES_MIN": worst["RAMAS TOTALES"],
        })

        for label, row in [("MAX", best), ("MIN", worst)]:
            plot_rows.append({
                "CAMPAÑA": camp,
                "EXTREMO": label,
                "TURNO": row["TURNO"],
                "KG/HA_pond": row["KG/HA_pond"],
                "MADERAS PRINCIPALES": row["MADERAS PRINCIPALES"],
                "CARGADORES": row["CARGADORES"],
                "RAMAS TOTALES": row["RAMAS TOTALES"],
            })

    out_table = pd.DataFrame(out_rows).sort_values("CAMPAÑA")
    plot_df = pd.DataFrame(plot_rows).sort_values(["CAMPAÑA", "EXTREMO"])
    return out_table, plot_df


def construir_modelo_importancia(df_model: pd.DataFrame, target_col: str):
    """
    Modelo: RandomForestRegressor
    - Numéricas: imputación mediana
    - Categóricas: imputación moda + OneHot
    - Importancias: feature_importances_ y luego agregación por variable original
    """
    drop_cols = [target_col, "FECHA PODA", "FECHA FIN DE SIEMBRA"]
    X = df_model.drop(columns=[c for c in drop_cols if c in df_model.columns], errors="ignore")
    y = df_model[target_col]

    cat_cols = [c for c in X.columns if (c in CATEGORICAS_BASE) or (X[c].dtype in ["string", "object"])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop"
    )

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=5
    )

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pipe.fit(X_train, y_train)

    pre = pipe.named_steps["preprocessor"]
    feature_names = []

    if len(num_cols) > 0:
        feature_names.extend(num_cols)

    if len(cat_cols) > 0:
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(list(ohe.get_feature_names_out(cat_cols)))

    importances = pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})

    def base_var(f):
        for c in cat_cols:
            if f.startswith(c + "_"):
                return c
        return f

    imp_df["base_feature"] = imp_df["feature"].apply(base_var)
    imp_agg = (
        imp_df.groupby("base_feature", as_index=False)["importance"].sum()
        .sort_values("importance", ascending=False)
    )

    return imp_agg


# -------------------------
# UI: SUBIR ARCHIVO
# -------------------------
st.sidebar.header("📁 Cargar Excel")

uploaded_file = st.sidebar.file_uploader(
    "Sube el Excel consolidado (.xlsx)", type=["xlsx"]
)

st.sidebar.caption("Requisito: la hoja debe llamarse exactamente **DATA**.")

if uploaded_file is None:
    st.info("📌 Sube tu archivo Excel para empezar.")
    st.stop()

with st.spinner("Leyendo Excel..."):
    df_raw = cargar_excel(uploaded_file, HOJA)
    validar_columnas(df_raw)
    df = preparar_df(df_raw)

# Filtrado a campañas objetivo (2022-2025)
if "CAMPAÑA" in df.columns:
    df = df[df["CAMPAÑA"].between(2022, 2025, inclusive="both")]

# Validación mínima
for must in ["CAMPAÑA", "SEMANA", "KG/HA", "Ha COSECHADA"]:
    if must not in df.columns:
        st.error(f"Falta columna obligatoria para este dashboard: '{must}'")
        st.stop()

df = df[df["KG/HA"].notna()].copy()
df = df[df["Ha COSECHADA"].notna()].copy()

if df.empty:
    st.warning("No hay datos con KG/HA y Ha COSECHADA válidos.")
    st.stop()


# -------------------------
# FILTROS
# -------------------------
st.sidebar.header("🎛️ Filtros")

def _opts(col):
    if col not in df.columns:
        return []
    return sorted([x for x in df[col].dropna().unique().tolist()])

camp_opts = _opts("CAMPAÑA")
fundo_opts = _opts("FUNDO")
etapa_opts = _opts("ETAPA")
campo_opts = _opts("CAMPO")
turno_opts = _opts("TURNO")
var_opts = _opts("VARIEDAD")
edadf_opts = _opts("EDAD PLANTA FINAL")
siembra_opts = _opts("SIEMBRA")
poda_opts = _opts("TIPO PODA")

camp_sel = st.sidebar.multiselect("CAMPAÑA", options=camp_opts, default=camp_opts)
fundo_sel = st.sidebar.multiselect("FUNDO", options=fundo_opts, default=fundo_opts)
etapa_sel = st.sidebar.multiselect("ETAPA", options=etapa_opts, default=etapa_opts)
campo_sel = st.sidebar.multiselect("CAMPO", options=campo_opts, default=campo_opts)
turno_sel = st.sidebar.multiselect("TURNO", options=turno_opts, default=turno_opts)
var_sel = st.sidebar.multiselect("VARIEDAD", options=var_opts, default=var_opts)

edadf_sel = st.sidebar.multiselect("EDAD PLANTA FINAL", options=edadf_opts, default=edadf_opts)
siembra_sel = st.sidebar.multiselect("SIEMBRA (Bolsa/Maceta/Suelo)", options=siembra_opts, default=siembra_opts)
poda_sel = st.sidebar.multiselect("TIPO PODA", options=poda_opts, default=poda_opts)

if "SEMANA" in df.columns and df["SEMANA"].notna().any():
    smin = int(df["SEMANA"].dropna().min())
    smax = int(df["SEMANA"].dropna().max())
    sem_rango = st.sidebar.slider("Rango de SEMANA", min_value=smin, max_value=smax, value=(smin, smax))
else:
    sem_rango = None

filtros = {
    "CAMPAÑA": camp_sel,
    "FUNDO": fundo_sel,
    "ETAPA": etapa_sel,
    "CAMPO": campo_sel,
    "TURNO": turno_sel,
    "VARIEDAD": var_sel,
    "EDAD PLANTA FINAL": edadf_sel,
    "SIEMBRA": siembra_sel,
    "TIPO PODA": poda_sel,
}
if sem_rango is not None:
    filtros["SEMANA"] = sem_rango

df_f = filtrar_df(df, filtros)

if df_f.empty:
    st.warning("Con estos filtros no hay datos.")
    st.stop()


# ============================================================
# 1) TABLA RESUMEN POR CAMPAÑA (reemplaza KPIs)
# ============================================================
st.subheader("Resumen por campaña (ponderado por Ha COSECHADA)")
tabla_camp = agg_campaign_table(df_f)

# formato
tabla_show = tabla_camp.copy()
for c in ["KG", "KG/HA (pond Ha COSECHADA)", "PESO (g) (pond Ha COSECHADA)", "CALIBRE (mm) (pond Ha COSECHADA)"]:
    if c in tabla_show.columns:
        tabla_show[c] = tabla_show[c].astype(float).round(2)

st.dataframe(tabla_show, use_container_width=True, height=220)

# Opcional: gráfico rápido (KG/HA pond por campaña)
fig_camp = px.bar(tabla_camp, x="CAMPAÑA", y="KG/HA (pond Ha COSECHADA)",
                  title="KG/HA ponderado por Ha COSECHADA (por campaña)")
st.plotly_chart(fig_camp, use_container_width=True)

st.divider()


# ============================================================
# 8) SCATTER (tipo imagen 11): X vs KG/HA ponderado por nivel
# ============================================================
st.subheader("Dispersión (X vs KG/HA ponderado) + BEST/WORST")

# Variables X numéricas elegibles (evitar categóricas)
cand_x = [
    "FLORES", "FRUTO MADURO", "PESO BAYA (g)", "CALIBRE BAYA (mm)",
    "DENSIDAD", "Ha COSECHADA", "kilogramos",
    "MADERAS PRINCIPALES", "CARGADORES", "RAMAS TOTALES", "TERMINALES",
    "ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT",
] + BROTES_CORE
cand_x = [c for c in cand_x if c in df_f.columns]

if len(cand_x) == 0:
    st.warning("No hay variables numéricas disponibles para el scatter con estos filtros.")
else:
    col_sc1, col_sc2 = st.columns([1, 3])

    with col_sc1:
        nivel = st.selectbox("Nivel de agregación", ["TURNO", "CAMPO", "VARIEDAD"], index=0)
        x_var = st.selectbox("Variable X", cand_x, index=0)

    # Definir group_cols según nivel
    if nivel == "TURNO":
        group_cols = ["CAMPAÑA", "VARIEDAD", "TURNO"]
    elif nivel == "CAMPO":
        group_cols = ["CAMPAÑA", "VARIEDAD", "CAMPO"]
    else:
        group_cols = ["CAMPAÑA", "VARIEDAD"]

    # Agregación ponderada
    rows = []
    for keys, g in df_f.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["KG/HA_pond"] = wavg(g["KG/HA"], g["Ha COSECHADA"])
        row[x_var] = wavg(g[x_var], g["Ha COSECHADA"])
        rows.append(row)

    sc = pd.DataFrame(rows).dropna(subset=["KG/HA_pond", x_var])

    if sc.empty:
        st.warning("No hay puntos suficientes para el scatter en este nivel.")
    else:
        # best / worst global en el nivel elegido
        idx_best = sc["KG/HA_pond"].idxmax()
        idx_worst = sc["KG/HA_pond"].idxmin()
        sc["POINT"] = "NORMAL"
        sc.loc[idx_best, "POINT"] = "BEST"
        sc.loc[idx_worst, "POINT"] = "WORST"

        with col_sc2:
            fig_sc = px.scatter(
                sc,
                x=x_var, y="KG/HA_pond",
                color="POINT",
                hover_data=group_cols + ["KG/HA_pond", x_var],
                title=f"{x_var} vs KG/HA (ponderado por Ha COSECHADA) | Nivel: {nivel}"
            )
            st.plotly_chart(fig_sc, use_container_width=True)

st.divider()


# ============================================================
# 2) MAX/MIN por TURNO dentro de (CAMPAÑA + VARIEDAD)
#    -> NO APLICA si TURNO fue fijado (no todos seleccionados)
# ============================================================
st.subheader("Best vs Worst TURNO dentro de (CAMPAÑA + VARIEDAD)")

turno_fijado = (len(turno_sel) < len(turno_opts))  # si eligió menos que todo, consideramos fijado
if turno_fijado:
    st.warning("No aplica: ya fijaste TURNO en los filtros.")
else:
    if len(var_sel) != 1:
        st.info("Para esta vista, selecciona **1 sola VARIEDAD** en el filtro (VARIEDAD).")
    else:
        variedad_obj = var_sel[0]
        out_table, plot_df = turnos_max_min_por_var_camp(df_f, variedad_obj)

        if out_table is None or out_table.empty:
            st.warning("No se pudo calcular MAX/MIN por TURNO para esa variedad con los filtros actuales.")
        else:
            st.markdown(f"**VARIEDAD:** {variedad_obj}")
            st.dataframe(out_table, use_container_width=True)

            # Gráfico estilo “barras + líneas” por campaña (MAX y MIN)
            # Barras: KG/HA_pond, Líneas: estructura
            for camp in sorted(plot_df["CAMPAÑA"].dropna().unique().tolist()):
                sub = plot_df[plot_df["CAMPAÑA"] == camp].copy()
                if sub.empty:
                    continue

                x_labels = [f"{variedad_obj}<br>{r['EXTREMO']}<br>{r['TURNO']}" for _, r in sub.iterrows()]

                fig = go.Figure()

                # barra kg/ha
                fig.add_trace(go.Bar(
                    x=x_labels,
                    y=sub["KG/HA_pond"],
                    name="KG/HA (pond)",
                ))

                # líneas estructura (eje secundario)
                fig.add_trace(go.Scatter(
                    x=x_labels, y=sub["MADERAS PRINCIPALES"], mode="lines+markers",
                    name="MADERAS PRINCIPALES", yaxis="y2"
                ))
                fig.add_trace(go.Scatter(
                    x=x_labels, y=sub["CARGADORES"], mode="lines+markers",
                    name="CARGADORES", yaxis="y2"
                ))
                fig.add_trace(go.Scatter(
                    x=x_labels, y=sub["RAMAS TOTALES"], mode="lines+markers",
                    name="RAMAS TOTALES", yaxis="y2"
                ))

                fig.update_layout(
                    title=f"KG/HA MAX vs MIN (por TURNO) + Estructura | CAMPAÑA {camp}",
                    xaxis_title="Extremos por turno",
                    yaxis_title="KG/HA (ponderado)",
                    yaxis2=dict(title="Estructura (pond)", overlaying="y", side="right"),
                    legend=dict(orientation="h"),
                    bargap=0.35
                )

                st.plotly_chart(fig, use_container_width=True)

st.divider()


# ============================================================
# 3) Curva semanal KG/HA ponderado por Ha COSECHADA
# ============================================================
st.subheader("Curva semanal de KG/HA (ponderado por Ha COSECHADA)")

weekly = agg_weekly(df_f)
weekly = weekly.dropna(subset=["SEMANA", "KG/HA_pond"])

if weekly.empty:
    st.warning("No hay datos suficientes para curva semanal.")
else:
    fig_week = px.line(
        weekly, x="SEMANA", y="KG/HA_pond", color="CAMPAÑA",
        markers=True, title="KG/HA ponderado por semana (según filtros)"
    )
    st.plotly_chart(fig_week, use_container_width=True)

st.divider()


# ============================================================
# 4) SIEMBRA y EDAD FINAL (ambos ponderados) +  KG/PLANTA vs campañas
# ============================================================
st.subheader("Comparaciones ponderadas (Ha COSECHADA): SIEMBRA, EDAD PLANTA FINAL y KG/PLANTA")

cA, cB = st.columns(2)

with cA:
    if "SIEMBRA" in df_f.columns:
        si = agg_cat_weighted(df_f, "SIEMBRA", "KG/HA")
        fig_si = px.bar(si, x="SIEMBRA", y="KG/HA_pond", title="KG/HA ponderado por SIEMBRA")
        st.plotly_chart(fig_si, use_container_width=True)
    else:
        st.warning("No existe SIEMBRA en la data.")

with cB:
    if "EDAD PLANTA FINAL" in df_f.columns:
        ed = agg_cat_weighted(df_f, "EDAD PLANTA FINAL", "KG/HA")
        # orden 1,2,3+
        orden = ["1", "2", "3+"]
        ed["EDAD PLANTA FINAL"] = ed["EDAD PLANTA FINAL"].astype("string")
        ed["ord"] = ed["EDAD PLANTA FINAL"].apply(lambda x: orden.index(x) if x in orden else 99)
        ed = ed.sort_values("ord")
        fig_ed = px.bar(ed, x="EDAD PLANTA FINAL", y="KG/HA_pond", title="KG/HA ponderado por EDAD PLANTA FINAL")
        st.plotly_chart(fig_ed, use_container_width=True)
    else:
        st.warning("No existe EDAD PLANTA FINAL en la data.")

# KG/PLANTA vs campañas (ponderado por Ha COSECHADA)
if "KG/PLANTA" in df_f.columns and df_f["KG/PLANTA"].notna().any():
    rows = []
    for camp, g in df_f.groupby("CAMPAÑA", dropna=False):
        rows.append({
            "CAMPAÑA": camp,
            "KG/PLANTA_pond": wavg(g["KG/PLANTA"], g["Ha COSECHADA"])
        })
    kgp = pd.DataFrame(rows).sort_values("CAMPAÑA").dropna()
    fig_kgp = px.line(kgp, x="CAMPAÑA", y="KG/PLANTA_pond", markers=True, title="KG/PLANTA ponderado (Ha COSECHADA) vs campañas")
    st.plotly_chart(fig_kgp, use_container_width=True)

st.divider()


# ============================================================
# 5) VARIEDAD: ranking ponderado + VS por campañas (heatmap)
# ============================================================
st.subheader("Variedades: ranking (KG/HA ponderado) + VS por campañas")

cV1, cV2 = st.columns([1, 2])

with cV1:
    topN = st.slider("Top N variedades por frecuencia", 5, 25, 10)
    var_rank = agg_variety_rank(df_f, topN=topN)
    st.dataframe(var_rank, use_container_width=True, height=520)

with cV2:
    fig_v = px.bar(
        var_rank.sort_values("KG/HA_pond", ascending=True),
        x="KG/HA_pond", y="VARIEDAD", orientation="h",
        title="Promedio KG/HA ponderado (Top variedades por frecuencia)"
    )
    st.plotly_chart(fig_v, use_container_width=True)

# VS por campaña (heatmap)
hm = agg_variety_vs_campaign(df_f, topN_vars=min(15, max(10, topN)))
if not hm.empty:
    fig_hm = px.density_heatmap(
        hm, x="CAMPAÑA", y="VARIEDAD", z="KG/HA_pond",
        title="VS: VARIEDAD x CAMPAÑA (KG/HA ponderado)",
        histfunc="avg"
    )
    st.plotly_chart(fig_hm, use_container_width=True)

st.divider()


# ============================================================
# 6) CORRELACIONES (solo heatmap y solo columnas indicadas)
# ============================================================
st.subheader("Mapa de correlaciones (solo variables seleccionadas)")

corr_cols = [
    "KG/HA", "kilogramos", "FLORES", "Ha COSECHADA", "DENSIDAD", "FRUTO MADURO",
    "PESO BAYA (g)", "CALIBRE BAYA (mm)", "SEMANA DE SIEMBRA",
    "MADERAS PRINCIPALES", "CARGADORES", "RAMAS TOTALES", "TERMINALES", "EDAD PLANTA"
]
corr_cols = [c for c in corr_cols if c in df_f.columns]

df_corr = df_f[corr_cols].copy()
for c in df_corr.columns:
    df_corr[c] = pd.to_numeric(df_corr[c], errors="coerce")

# Mantener columnas con data suficiente
valid_cols = [c for c in df_corr.columns if df_corr[c].notna().mean() >= 0.20]
df_corr = df_corr[valid_cols]

if df_corr.shape[1] < 3:
    st.warning("No hay suficientes columnas con data para correlación (con estos filtros).")
else:
    corr = df_corr.corr(numeric_only=True)
    fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns))
    fig_corr.update_layout(title="Mapa de correlaciones (variables con data suficiente)")
    st.plotly_chart(fig_corr, use_container_width=True)

st.divider()


# ============================================================
# 7) MODELO (IMPORTANCIAS) - se mantiene
# ============================================================
st.subheader("Variables más asociadas a KG/HA (modelo)")

df_model = df_f.copy()
cols_model = list(dict.fromkeys(
    ["KG/HA", "AÑO", "CAMPAÑA", "SEMANA"] +
    CATEGORICAS_BASE +
    ["kilogramos", "Ha TURNO", "Ha COSECHADA", "DENSIDAD", "KG/PLANTA"] +
    ESTRUCTURA_CORE + FENOLOGIA_CORE + BROTES_CORE
))
cols_model = [c for c in cols_model if c in df_model.columns]
df_model = df_model[cols_model].copy()

# Drop columnas ultra-sparse
drop_sparse = [c for c in df_model.columns if c != "KG/HA" and df_model[c].notna().mean() < 0.01]
if drop_sparse:
    df_model = df_model.drop(columns=drop_sparse)

if len(df_model) < 200:
    st.warning(f"No entreno modelo porque hay pocos registros ({len(df_model)}). Amplía filtros.")
else:
    with st.spinner("Entrenando modelo e importancias..."):
        imp_agg = construir_modelo_importancia(df_model, target_col="KG/HA")

    topk = st.slider("Top K variables (importancia)", 10, 40, 20)
    imp_show = imp_agg.head(topk)

    cM1, cM2 = st.columns([1, 2])
    with cM1:
        st.dataframe(imp_show, use_container_width=True, height=520)

    with cM2:
        fig_imp = px.bar(
            imp_show.sort_values("importance"),
            x="importance", y="base_feature", orientation="h",
            title="Importancia global (agregada por variable original)"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
