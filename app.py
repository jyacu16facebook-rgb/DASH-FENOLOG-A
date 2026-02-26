# app.py
# ============================================================
# DASHBOARD: Drivers fenológicos/estructura asociados a KG/HA
# Fuente: "CONSOLIDADO 2022-2026.xlsx" | Hoja: "DATA"
# Autor: (tu proyecto)
# ============================================================

import os
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
st.set_page_config(
    page_title="Fenología vs Rendimiento (KG/HA)",
    layout="wide"
)

st.title("🫐 Fenología y estructura vs rendimiento (KG/HA) | Campañas 2022–2025")
st.caption("Objetivo: identificar variables fenológicas/estructura más asociadas a cambios en KG/HA y facilitar el análisis por filtros (Fundo→Etapa→Campo→Turno→Variedad→Semana→Campaña).")


# -------------------------
# PARAMETROS DATA
# -------------------------
RUTA_DEFAULT = r"C:\Users\JeinerJhoelLunaYacup\Desktop\CONSOLIDADO 2022-2026.xlsx"
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

# Variables “core” para mostrar en la tarjeta de MAX/MIN:
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
    """Convierte a numérico tolerando strings raros."""
    return pd.to_numeric(s, errors="coerce")


def _normalize_edad_final(x):
    """Normaliza EDAD PLANTA FINAL a {1,2,'3+'} si viniera con formatos distintos."""
    if pd.isna(x):
        return np.nan
    x_str = str(x).strip()
    # acepta "3+", "3 +", "3", "3.0", etc
    if re.match(r"^3\s*\+?$", x_str) or "3+" in x_str:
        return "3+"
    if x_str in ["1", "1.0", "01"]:
        return "1"
    if x_str in ["2", "2.0", "02"]:
        return "2"
    # si ya viene correcto:
    if x_str in ["3+"]:
        return "3+"
    return x_str


def _smart_nan_for_zeros(df: pd.DataFrame, cols: list[str], zero_is_missing: bool = True) -> pd.DataFrame:
    """
    Muchas variables no evaluadas pueden venir en 0.
    Criterio: si una columna tiene demasiados ceros (y muy pocos valores >0),
    interpretamos que 0 podría ser 'no evaluado' y lo convertimos a NaN.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if not pd.api.types.is_numeric_dtype(out[c]):
            continue

        s = out[c]
        # porcentaje de ceros en no-nulos
        nn = s.notna().sum()
        if nn == 0:
            continue

        zeros = (s == 0).sum()
        pos = (s > 0).sum()

        # Heurística:
        # - si casi todo son ceros y casi no hay positivos, es sospechoso (no evaluado)
        # - si hay muchos positivos, el cero sí puede ser valor real (ej. conteo 0)
        if zero_is_missing and zeros / nn > 0.85 and pos / nn < 0.10:
            out.loc[out[c] == 0, c] = np.nan

    return out


@st.cache_data(show_spinner=False)
def cargar_data(ruta: str, hoja: str) -> pd.DataFrame:
    df = pd.read_excel(ruta, sheet_name=hoja)
    # Limpieza básica de columnas: quitar espacios extremos
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validar_columnas(df: pd.DataFrame):
    faltantes = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
    if faltantes:
        st.warning("⚠️ Columnas esperadas NO encontradas (revisa nombres exactos en tu Excel):")
        st.write(faltantes)


def preparar_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Tipos:
    for c in ["AÑO", "CAMPAÑA", "SEMANA"]:
        if c in d.columns:
            d[c] = _to_numeric_safe(d[c]).astype("Int64")

    # Categóricas a string
    for c in CATEGORICAS_BASE:
        if c in d.columns:
            d[c] = d[c].astype("string").str.strip()

    # Normalizar EDAD PLANTA FINAL
    if "EDAD PLANTA FINAL" in d.columns:
        d["EDAD PLANTA FINAL"] = d["EDAD PLANTA FINAL"].apply(_normalize_edad_final).astype("string")

    # Fechas (si vienen como texto)
    for c in ["FECHA PODA", "FECHA FIN DE SIEMBRA"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")

    # Numéricas: todo lo que no sea base categórica y no sea llaves
    llaves = ["AÑO", "CAMPAÑA", "SEMANA"] + CATEGORICAS_BASE + ["EDAD PLANTA"]
    num_cols = [c for c in d.columns if c not in llaves and c not in ["FECHA PODA", "FECHA FIN DE SIEMBRA"]]

    # Fuerza a numérico donde se pueda (respetando nombres)
    for c in num_cols:
        d[c] = _to_numeric_safe(d[c])

    # Si KG/HA existe, úsala; si no, calcula proxy con kilogramos / Ha TURNO
    if "KG/HA" not in d.columns:
        d["KG/HA"] = np.nan

    # Asegurar kilogramos, Ha TURNO
    if "kilogramos" in d.columns and "Ha TURNO" in d.columns:
        mask_nan = d["KG/HA"].isna()
        denom = d["Ha TURNO"].replace({0: np.nan})
        d.loc[mask_nan, "KG/HA"] = d.loc[mask_nan, "kilogramos"] / denom

    # Crear KG/PLANTA (si densidad disponible)
    if "DENSIDAD" in d.columns and "Ha TURNO" in d.columns and "kilogramos" in d.columns:
        plantas = d["Ha TURNO"].replace({0: np.nan}) * d["DENSIDAD"].replace({0: np.nan})
        d["KG/PLANTA"] = d["kilogramos"] / plantas
    else:
        d["KG/PLANTA"] = np.nan

    # Heurística de ceros como missing para variables típicamente "no evaluadas" en algunos periodos
    candidatos_zeros_missing = (
        ESTRUCTURA_CORE + BROTES_CORE + ["ALTURA_PLANTA_ULT", "ANCHO_PLANTA_ULT",
                                         "PESO BAYA (g)", "PESO BAYA CREMOSO (g)",
                                         "CALIBRE BAYA (mm)", "CALIBRE CREMOSO (mm)"]
    )
    d = _smart_nan_for_zeros(d, candidatos_zeros_missing, zero_is_missing=True)

    return d


def filtrar_df(d: pd.DataFrame, filtros: dict) -> pd.DataFrame:
    out = d.copy()
    for col, val in filtros.items():
        if val is None:
            continue
        if col not in out.columns:
            continue
        if isinstance(val, tuple) and len(val) == 2:
            # rango para SEMANA
            lo, hi = val
            out = out[(out[col].notna()) & (out[col] >= lo) & (out[col] <= hi)]
        elif isinstance(val, list):
            if len(val) > 0:
                out = out[out[col].isin(val)]
        else:
            out = out[out[col] == val]
    return out


def resumen_registro(row: pd.Series) -> pd.DataFrame:
    """
    Construye una tabla vertical (Variable, Valor) para mostrar el caso MAX/MIN.
    """
    campos = [
        "AÑO", "CAMPAÑA", "SEMANA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
        "SIEMBRA", "TIPO PODA", "EDAD PLANTA FINAL",
        "kilogramos", "Ha TURNO", "Ha COSECHADA", "KG/HA", "KG/PLANTA", "DENSIDAD"
    ] + ESTRUCTURA_CORE + FENOLOGIA_CORE + BROTES_CORE

    data = []
    for c in campos:
        if c in row.index:
            v = row[c]
            # formateo
            if isinstance(v, (np.floating, float)) and not np.isnan(v):
                v_show = float(v)
            elif pd.isna(v):
                v_show = np.nan
            else:
                v_show = v
            data.append((c, v_show))
    return pd.DataFrame(data, columns=["Variable", "Valor"])


def construir_modelo_importancia(df_model: pd.DataFrame, target_col: str):
    """
    Modelo simple (RandomForest) con OneHot para categóricas.
    Devuelve importancia global de variables (post one-hot) agregada por variable original.
    """
    # Features candidatas: todas excepto fechas y target
    drop_cols = [target_col, "FECHA PODA", "FECHA FIN DE SIEMBRA"]
    X = df_model.drop(columns=[c for c in drop_cols if c in df_model.columns], errors="ignore")
    y = df_model[target_col]

    # Detectar categóricas y numéricas
    cat_cols = [c for c in X.columns if (c in CATEGORICAS_BASE) or (X[c].dtype == "string") or (X[c].dtype == "object")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Preprocesamiento
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

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
        max_depth=None,
        min_samples_leaf=5
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Split (para evitar overfit extremo, aunque aquí solo buscamos importancia)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    pipe.fit(X_train, y_train)

    # Sacar nombres de features post onehot
    pre = pipe.named_steps["preprocessor"]
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]

    # feature names
    feature_names = []
    if len(num_cols) > 0:
        feature_names.extend(num_cols)

    if len(cat_cols) > 0:
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names.extend(cat_feature_names)

    importances = pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})

    # Agregar a variable original (antes del onehot)
    def base_var(f):
        # ejemplo: "VARIEDAD_SOMETHING" -> "VARIEDAD"
        for c in cat_cols:
            if f.startswith(c + "_"):
                return c
        return f

    imp_df["base_feature"] = imp_df["feature"].apply(base_var)
    imp_agg = imp_df.groupby("base_feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False)

    return imp_agg, pipe


# -------------------------
# SIDEBAR: RUTA Y CARGA
# -------------------------
st.sidebar.header("📁 Fuente de datos")

ruta = st.sidebar.text_input("Ruta del Excel", value=RUTA_DEFAULT)
hoja = st.sidebar.text_input("Hoja", value=HOJA)

if not os.path.exists(ruta):
    st.error("No encuentro el archivo en esa ruta. Revisa la ruta o mueve el Excel al lugar correcto.")
    st.stop()

with st.spinner("Leyendo Excel..."):
    df_raw = cargar_data(ruta, hoja)
    validar_columnas(df_raw)
    df = preparar_df(df_raw)

# Filtrado a campañas objetivo (2022–2025) si existe CAMPAÑA
if "CAMPAÑA" in df.columns:
    df = df[df["CAMPAÑA"].between(2022, 2025, inclusive="both")]

# -------------------------
# SIDEBAR: FILTROS
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

# Semana rango
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

# -------------------------
# VALIDACIONES TARGET
# -------------------------
if df_f.empty:
    st.warning("Con estos filtros no hay datos.")
    st.stop()

if "KG/HA" not in df_f.columns:
    st.error("No existe la columna 'KG/HA' y no se pudo calcular. Revisa 'kilogramos' y 'Ha TURNO'.")
    st.stop()

# quitamos registros sin target
df_f = df_f[df_f["KG/HA"].notna()]

if df_f.empty:
    st.warning("No hay datos con KG/HA válido en estos filtros.")
    st.stop()


# -------------------------
# KPIs
# -------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Registros filtrados", f"{len(df_f):,}".replace(",", "."))
kpi2.metric("Promedio KG/HA", f"{df_f['KG/HA'].mean():.2f}")
kpi3.metric("Máximo KG/HA", f"{df_f['KG/HA'].max():.2f}")
kpi4.metric("Mínimo KG/HA", f"{df_f['KG/HA'].min():.2f}")

st.divider()

# -------------------------
# PANEL MAX / MIN + EXPLICACIÓN ESTRUCTURA/FENOLOGIA
# -------------------------
st.subheader("🔎 Casos extremos (MAX / MIN KG/HA) + su estructura/fenología")

colA, colB = st.columns(2)

# Max
idx_max = df_f["KG/HA"].idxmax()
row_max = df_f.loc[idx_max]
max_table = resumen_registro(row_max)

with colA:
    st.markdown("### ✅ Caso MAX KG/HA")
    st.write(f"**KG/HA = {float(row_max['KG/HA']):.2f}**")
    st.dataframe(max_table, use_container_width=True, height=480)

# Min
idx_min = df_f["KG/HA"].idxmin()
row_min = df_f.loc[idx_min]
min_table = resumen_registro(row_min)

with colB:
    st.markdown("### ⚠️ Caso MIN KG/HA")
    st.write(f"**KG/HA = {float(row_min['KG/HA']):.2f}**")
    st.dataframe(min_table, use_container_width=True, height=480)

st.info(
    "Tip de lectura: compara en MAX vs MIN especialmente "
    "CARGADORES, TERMINALES, MADERAS PRINCIPALES, RAMAS TOTALES, "
    "FLORES/CUAJA/VERDES y PESO/CALIBRE, y además SIEMBRA y EDAD PLANTA FINAL."
)

st.divider()

# -------------------------
# TENDENCIA SEMANAL
# -------------------------
st.subheader("📈 Curva semanal de KG/HA (y comparación por campaña)")

if "SEMANA" in df_f.columns and "CAMPAÑA" in df_f.columns:
    # Agregación robusta
    agg = (df_f
           .groupby(["CAMPAÑA", "SEMANA"], dropna=False)["KG/HA"]
           .mean()
           .reset_index()
           .sort_values(["CAMPAÑA", "SEMANA"]))

    fig = px.line(
        agg, x="SEMANA", y="KG/HA", color="CAMPAÑA",
        markers=True,
        title="Promedio KG/HA por semana (según filtros)"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No puedo graficar curva semanal: faltan columnas SEMANA o CAMPAÑA.")

st.divider()

# -------------------------
# DISTRIBUCIÓN + EFECTO SIEMBRA / EDAD
# -------------------------
st.subheader("📊 Distribución de KG/HA y comparaciones clave (SIEMBRA, EDAD PLANTA FINAL)")

c1, c2 = st.columns(2)

with c1:
    fig_hist = px.histogram(
        df_f, x="KG/HA", nbins=40,
        title="Distribución de KG/HA (según filtros)"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    if "SIEMBRA" in df_f.columns:
        fig_box = px.box(
            df_f, x="SIEMBRA", y="KG/HA",
            title="KG/HA por SIEMBRA (Bolsa/Maceta/Suelo)"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("No existe SIEMBRA en la data filtrada.")

c3, c4 = st.columns(2)

with c3:
    if "EDAD PLANTA FINAL" in df_f.columns:
        fig_box2 = px.box(
            df_f, x="EDAD PLANTA FINAL", y="KG/HA",
            title="KG/HA por EDAD PLANTA FINAL (1,2,3+)"
        )
        st.plotly_chart(fig_box2, use_container_width=True)
    else:
        st.warning("No existe EDAD PLANTA FINAL.")

with c4:
    if "VARIEDAD" in df_f.columns:
        topN = st.slider("Top N variedades por frecuencia", 5, 25, 10)
        top_vars = df_f["VARIEDAD"].value_counts().head(topN).index.tolist()
        tmp = df_f[df_f["VARIEDAD"].isin(top_vars)]
        fig_bar = px.bar(
            tmp.groupby("VARIEDAD")["KG/HA"].mean().reset_index().sort_values("KG/HA", ascending=False),
            x="VARIEDAD", y="KG/HA",
            title="Promedio KG/HA por VARIEDAD (Top por frecuencia)"
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("No existe VARIEDAD.")

st.divider()

# -------------------------
# CORRELACIONES (solo numéricas) - con criterio de vacíos
# -------------------------
st.subheader("🧠 Señales numéricas rápidas (correlaciones)")

# Solo numéricas (incluye fenología/estructura/brotes) con mínimo de datos
num_candidates = [
    "KG/HA", "kilogramos", "Ha TURNO", "Ha COSECHADA", "DENSIDAD", "KG/PLANTA"
] + FENOLOGIA_CORE + ESTRUCTURA_CORE + BROTES_CORE

num_candidates = [c for c in num_candidates if c in df_f.columns]

df_num = df_f[num_candidates].copy()
# quedarse con columnas con suficiente data
valid_cols = []
for c in df_num.columns:
    if df_num[c].notna().mean() >= 0.20:  # al menos 20% con datos
        valid_cols.append(c)

df_num = df_num[valid_cols]

if df_num.shape[1] >= 3:
    corr = df_num.corr(numeric_only=True)
    # Top correlaciones absolutas con KG/HA
    if "KG/HA" in corr.columns:
        corr_kg = (corr["KG/HA"].drop("KG/HA").abs().sort_values(ascending=False).head(15))
        st.write("Top 15 correlaciones absolutas con **KG/HA** (solo orientación; no es causalidad):")
        st.dataframe(corr_kg.reset_index().rename(columns={"index": "Variable", "KG/HA": "|Corr|"}), use_container_width=True)

        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns
            )
        )
        fig_corr.update_layout(title="Mapa de correlaciones (variables numéricas con data suficiente)")
        st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.warning("No hay suficientes variables numéricas con datos para correlación (muchas vacías en estos filtros).")

st.divider()

# -------------------------
# MODELO: IMPORTANCIA DE VARIABLES (asociación global)
# -------------------------
st.subheader("🏗️ Variables más asociadas a KG/HA (modelo)")

st.caption(
    "Esto entrena un modelo con las variables disponibles (numéricas + categóricas) "
    "y calcula importancia global. Sirve para priorizar variables; no prueba causalidad."
)

# Preparar data model: evitar filas sin target
df_model = df_f.copy()

# Reducir a columnas relevantes + llaves (evita ruido extremo, pero mantiene potencia)
cols_model = list(dict.fromkeys(
    ["KG/HA", "AÑO", "CAMPAÑA", "SEMANA"] +
    CATEGORICAS_BASE +
    ["kilogramos", "Ha TURNO", "Ha COSECHADA", "DENSIDAD", "KG/PLANTA"] +
    ESTRUCTURA_CORE + FENOLOGIA_CORE + BROTES_CORE
))
cols_model = [c for c in cols_model if c in df_model.columns]
df_model = df_model[cols_model].copy()

# Criterio: si una columna tiene demasiados nulos, se queda igual (el imputador se encarga),
# pero si es 99% nula, no aporta.
drop_sparse = []
for c in df_model.columns:
    if c == "KG/HA":
        continue
    if df_model[c].notna().mean() < 0.01:
        drop_sparse.append(c)

if drop_sparse:
    df_model = df_model.drop(columns=drop_sparse)

min_rows_model = 200
if len(df_model) < min_rows_model:
    st.warning(f"No entreno modelo porque hay pocos registros ({len(df_model)}). "
               f"Amplía filtros o baja exigencia mínima en código.")
else:
    with st.spinner("Entrenando modelo e importancias..."):
        imp_agg, pipe = construir_modelo_importancia(df_model, target_col="KG/HA")

    topk = st.slider("Top K variables (importancia)", 10, 40, 20)
    imp_show = imp_agg.head(topk)

    cA, cB = st.columns([1, 2])

    with cA:
        st.dataframe(imp_show, use_container_width=True, height=520)

    with cB:
        fig_imp = px.bar(
            imp_show.sort_values("importance"),
            x="importance",
            y="base_feature",
            orientation="h",
            title="Importancia global (agregada por variable original)"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.success(
        "Listo. Usa este ranking para enfocar discusión: "
        "si aparecen SIEMBRA, EDAD PLANTA FINAL, CARGADORES, TERMINALES, FLORES/CUAJA, PESO/CALIBRE, etc., "
        "son candidatos fuertes a explicar diferencias de KG/HA en 2022–2025."
    )

# -------------------------
# EXPORT DE CASOS MAX/MIN (opcional)
# -------------------------
st.divider()
st.subheader("⬇️ Export rápido (casos MAX y MIN)")

export = st.checkbox("Generar Excel de MAX/MIN (en carpeta del app)", value=False)
if export:
    out_path = "casos_extremos_max_min_kg_ha.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        max_table.to_excel(writer, sheet_name="MAX_KG_HA", index=False)
        min_table.to_excel(writer, sheet_name="MIN_KG_HA", index=False)
    st.success(f"Generado: {out_path} (queda en la misma carpeta donde corres Streamlit).")
