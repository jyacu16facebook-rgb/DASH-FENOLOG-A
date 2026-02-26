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
st.caption(
    "Objetivo: identificar variables fenológicas/estructura más asociadas a cambios en KG/HA y facilitar el análisis por filtros "
    "(Fundo→Etapa→Campo→Turno→Variedad→Semana→Campaña)."
)

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
    if re.match(r"^3\s*\+?$", x_str) or "3+" in x_str:
        return "3+"
    if x_str in ["1", "1.0", "01"]:
        return "1"
    if x_str in ["2", "2.0", "02"]:
        return "2"
    if x_str in ["3+"]:
        return "3+"
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

    for c in ["AÑO", "CAMPAÑA", "SEMANA"]:
        if c in d.columns:
            d[c] = _to_numeric_safe(d[c]).astype("Int64")

    for c in CATEGORICAS_BASE:
        if c in d.columns:
            d[c] = d[c].astype("string").str.strip()

    if "EDAD PLANTA FINAL" in d.columns:
        d["EDAD PLANTA FINAL"] = d["EDAD PLANTA FINAL"].apply(_normalize_edad_final).astype("string")

    for c in ["FECHA PODA", "FECHA FIN DE SIEMBRA"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")

    llaves = ["AÑO", "CAMPAÑA", "SEMANA"] + CATEGORICAS_BASE + ["EDAD PLANTA"]
    num_cols = [c for c in d.columns if c not in llaves and c not in ["FECHA PODA", "FECHA FIN DE SIEMBRA"]]

    for c in num_cols:
        d[c] = _to_numeric_safe(d[c])

    # Si KG/HA viene vacío, intentamos calcularlo
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


def resumen_registro(row: pd.Series) -> pd.DataFrame:
    campos = [
        "AÑO", "CAMPAÑA", "SEMANA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
        "SIEMBRA", "TIPO PODA", "EDAD PLANTA FINAL",
        "kilogramos", "Ha TURNO", "Ha COSECHADA", "KG/HA", "KG/PLANTA", "DENSIDAD"
    ] + ESTRUCTURA_CORE + FENOLOGIA_CORE + BROTES_CORE

    data = []
    for c in campos:
        if c in row.index:
            v = row[c]
            data.append((c, v))
    return pd.DataFrame(data, columns=["Variable", "Valor"])


def construir_modelo_importancia(df_model: pd.DataFrame, target_col: str):
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
    imp_agg = imp_df.groupby("base_feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False)

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
    st.warning("📌 Sube tu archivo Excel para empezar.")
    st.stop()

with st.spinner("Leyendo Excel..."):
    df_raw = cargar_excel(uploaded_file, HOJA)
    validar_columnas(df_raw)
    df = preparar_df(df_raw)

# Filtrado a campañas objetivo
if "CAMPAÑA" in df.columns:
    df = df[df["CAMPAÑA"].between(2022, 2025, inclusive="both")]

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

if "KG/HA" not in df_f.columns:
    st.error("No existe la columna 'KG/HA' y no se pudo calcular. Revisa 'kilogramos' y 'Ha TURNO'.")
    st.stop()

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
# MAX / MIN
# -------------------------
st.subheader("🔎 Casos extremos (MAX / MIN KG/HA) + su estructura/fenología")

colA, colB = st.columns(2)

idx_max = df_f["KG/HA"].idxmax()
row_max = df_f.loc[idx_max]
max_table = resumen_registro(row_max)

with colA:
    st.markdown("### ✅ Caso MAX KG/HA")
    st.write(f"**KG/HA = {float(row_max['KG/HA']):.2f}**")
    st.dataframe(max_table, use_container_width=True, height=480)

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
# CURVA SEMANAL
# -------------------------
st.subheader("📈 Curva semanal de KG/HA (comparación por campaña)")

if "SEMANA" in df_f.columns and "CAMPAÑA" in df_f.columns:
    agg = (df_f.groupby(["CAMPAÑA", "SEMANA"], dropna=False)["KG/HA"].mean()
           .reset_index()
           .sort_values(["CAMPAÑA", "SEMANA"]))
    fig = px.line(agg, x="SEMANA", y="KG/HA", color="CAMPAÑA", markers=True,
                  title="Promedio KG/HA por semana (según filtros)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No puedo graficar curva semanal: faltan columnas SEMANA o CAMPAÑA.")

st.divider()

# -------------------------
# DISTRIBUCIÓN / SIEMBRA / EDAD
# -------------------------
st.subheader("📊 Distribución de KG/HA y comparaciones clave (SIEMBRA, EDAD PLANTA FINAL)")

c1, c2 = st.columns(2)
with c1:
    fig_hist = px.histogram(df_f, x="KG/HA", nbins=40, title="Distribución de KG/HA")
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    if "SIEMBRA" in df_f.columns:
        fig_box = px.box(df_f, x="SIEMBRA", y="KG/HA", title="KG/HA por SIEMBRA")
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("No existe SIEMBRA en la data filtrada.")

c3, c4 = st.columns(2)
with c3:
    if "EDAD PLANTA FINAL" in df_f.columns:
        fig_box2 = px.box(df_f, x="EDAD PLANTA FINAL", y="KG/HA", title="KG/HA por EDAD PLANTA FINAL")
        st.plotly_chart(fig_box2, use_container_width=True)

with c4:
    if "VARIEDAD" in df_f.columns:
        topN = st.slider("Top N variedades por frecuencia", 5, 25, 10)
        top_vars = df_f["VARIEDAD"].value_counts().head(topN).index.tolist()
        tmp = df_f[df_f["VARIEDAD"].isin(top_vars)]
        fig_bar = px.bar(
            tmp.groupby("VARIEDAD")["KG/HA"].mean().reset_index().sort_values("KG/HA", ascending=False),
            x="VARIEDAD", y="KG/HA", title="Promedio KG/HA por VARIEDAD (Top por frecuencia)"
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# -------------------------
# CORRELACIONES (orientación)
# -------------------------
st.subheader("🧠 Señales numéricas rápidas (correlaciones)")

num_candidates = [
    "KG/HA", "kilogramos", "Ha TURNO", "Ha COSECHADA", "DENSIDAD", "KG/PLANTA"
] + FENOLOGIA_CORE + ESTRUCTURA_CORE + BROTES_CORE
num_candidates = [c for c in num_candidates if c in df_f.columns]

df_num = df_f[num_candidates].copy()
valid_cols = [c for c in df_num.columns if df_num[c].notna().mean() >= 0.20]
df_num = df_num[valid_cols]

if df_num.shape[1] >= 3:
    corr = df_num.corr(numeric_only=True)
    if "KG/HA" in corr.columns:
        corr_kg = corr["KG/HA"].drop("KG/HA").abs().sort_values(ascending=False).head(15)
        st.write("Top 15 correlaciones absolutas con **KG/HA** (solo orientación; no es causalidad):")
        st.dataframe(corr_kg.reset_index().rename(columns={"index": "Variable", "KG/HA": "|Corr|"}), use_container_width=True)

        fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns))
        fig_corr.update_layout(title="Mapa de correlaciones (variables numéricas con data suficiente)")
        st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.warning("No hay suficientes variables numéricas con datos para correlación (muchas vacías en estos filtros).")

st.divider()

# -------------------------
# IMPORTANCIA MODELO
# -------------------------
st.subheader("🏗️ Variables más asociadas a KG/HA (modelo)")

df_model = df_f.copy()
cols_model = list(dict.fromkeys(
    ["KG/HA", "AÑO", "CAMPAÑA", "SEMANA"] +
    CATEGORICAS_BASE +
    ["kilogramos", "Ha TURNO", "Ha COSECHADA", "DENSIDAD", "KG/PLANTA"] +
    ESTRUCTURA_CORE + FENOLOGIA_CORE + BROTES_CORE
))
cols_model = [c for c in cols_model if c in df_model.columns]
df_model = df_model[cols_model].copy()

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

    cA, cB = st.columns([1, 2])
    with cA:
        st.dataframe(imp_show, use_container_width=True, height=520)

    with cB:
        fig_imp = px.bar(
            imp_show.sort_values("importance"),
            x="importance", y="base_feature", orientation="h",
            title="Importancia global (agregada por variable original)"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
