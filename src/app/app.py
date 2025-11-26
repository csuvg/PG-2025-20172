import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any

st.set_page_config(page_title="IVESVI — Índice de vulnerabilidad", layout="centered")

# --- Constantes / Configuración ------------------------------------------------
VARIABLES_APOYO_INSTITUCIONAL = [
    'HEC_DEPTO',
    'CANTIDAD_COMISARIAS',
    'CANTIDAD_ESTACIONES',
    'CANTIDAD_SUBESTACIONES',
    'CANTIDAD_JUZGADOS_VIOLENCIA_CONTRA_MUJ_FEMINICIDIO',
    'CANTIDAD_TRIBUNALES_SENTENCIA_VIOLENCIA_CONTRA_MUJ_FEMINICIDIO',
    'CANTIDAD_FISCALIAS_MUNICIPALES',
    'CANTIDAD_AGENCIAS_FISCALES',
    'CANTIDAD_CENTRO_ATENCION_PERMANENTE',
    'CANTIDAD_CENTRO_SALUD',
    'CANTIDAD_PUESTO_SALUD',
    'CANTIDAD_HOSPITAL'
]

i_apoyo_dict = {
    'guatemala': 0.21870998433762268,
    'quetzaltenango': 0.38266391188301935,
    'huehuetenango': 0.46573906674965215,
    'jalapa': 0.07506629880423664,
    'escuintla': 0.23922770536496527,
    'el progreso': 0.51333278854146,
    'sacatepequez': 0.19976263262000996,
    'chimaltenango': 0.276308388269306,
    'santa rosa': 0.30656491475242736,
    'solola': 0.5607169454238273,
    'totonicapan': 0.08665269082007541,
    'suchitepequez': 0.43108672566375694,
    'retalhuleu': 0.22927254341153633,
    'san marcos': 0.4188102854765258,
    'alta verapaz': 0.18316291003557614,
    'quiche': 0.31644210162617026,
    'baja verapaz': 0.26733679314997966,
    'peten': 0.17755530044912474,
    'izabal': 0.18623612865002712,
    'zacapa': 0.6349827559780027,
    'chiquimula': 0.24591420910464912,
    'jutiapa': 0.14135037277390097
}

# Riesgos por modelo (orden de clases esperado por predict_proba)
RIESGO_VICTIM = [3, 2, 4, 1]
RIESGO_SITUATION = [3, 1, 2, 4, 5]
RIESGO_RELATION = [1, 4, 2, 3]

# --- Helpers -------------------------------------------------------------------
@st.cache_resource
def load_label_encoders(path: str = "./dataset/label_encoders.pkl"):
    try:
        with open(path, "rb") as f:
            return joblib.load(f)
    except Exception as e:
        st.error("Error al cargar label encoders: " + str(e))
        return None

@st.cache_resource
def load_models(base_path: str = "./models/") -> Dict[str, Any]:
    models = {}
    try:
        with open(f"{base_path}gmm_model_no_kmodes_relation_top1.pkl", "rb") as f:
            models['relation'] = joblib.load(f)
        with open(f"{base_path}gmm_model_no_kmodes_situation_top0.pkl", "rb") as f:
            models['situation'] = joblib.load(f)
        with open(f"{base_path}gmm_model_no_kmodes_victimas_top0.pkl", "rb") as f:
            models['victim'] = joblib.load(f)
    except Exception as e:
        st.error("Error al cargar modelos: " + str(e))
    return models

def safe_transform(le, value):
    """Transforma con label encoder manejando valores desconocidos."""
    try:
        return int(le.transform([value])[0])
    except Exception:
        # intentamos buscar string normalizado (minusculas/strip)
        try:
            return int(le.transform([str(value).strip().lower()])[0])
        except Exception:
            raise ValueError(f"Valor no codificable por encoder: {value}")

def prepare_input_dataframe(raw: Dict[str, Any], label_encoders: Dict[str, Any]) -> pd.DataFrame:
    # Columnas de entrada cruda que usa el form
    VARIABLES = [
        "VIC_ALFAB","VIC_ES_INDIGENA","VIC_NIV_ESCOLARIDAD","VIC_TRABAJA","CANTIDAD_HIJOS",
        "AGR_ALFAB","AGR_ES_INDIGENA","AGR_NIV_ESCOLARIDAD","AGR_TRABAJA","AGR_SEXO",
        "VIOLENCIA_FISICA","VIOLENCIA_PSICOLOGICA","VIOLENCIA_SEXUAL","VIOLENCIA_PATRIMONIAL",
        "VIC_EST_CIV","VIC_REL_AGR","AGR_EST_CIV"
    ]
    df = pd.DataFrame([{k: raw.get(k) for k in VARIABLES}])
    # generar encoded cols
    for var in VARIABLES:
        enc_col = var + "_ENC"
        if label_encoders is None or var not in label_encoders:
            raise RuntimeError(f"No hay encoder para la variable {var}")
        le = label_encoders[var]
        df[enc_col] = df[var].apply(lambda v: safe_transform(le, v))
    return df

def score_from_probs(probs: np.ndarray, riesgo: list) -> float:
    s = 0.0
    for idx, p in enumerate(probs[0]):
        s += p * riesgo[idx]
    return s

def format4(x: float) -> str:
    return f"{x:.4f}"

# --- Predicción y presentación -------------------------------------------------
def gmm_full_predict(input_data: pd.DataFrame, depto: str, models: Dict[str, Any]):
    # arreglos de features según modelos
    X_eval_victim = input_data[["VIC_ALFAB_ENC","VIC_ES_INDIGENA_ENC","VIC_NIV_ESCOLARIDAD_ENC","VIC_TRABAJA_ENC","CANTIDAD_HIJOS_ENC"]].iloc[0].to_numpy().reshape(1, -1)
    X_eval_situation = input_data[["AGR_ALFAB_ENC","AGR_ES_INDIGENA_ENC","AGR_NIV_ESCOLARIDAD_ENC","AGR_SEXO_ENC","VIOLENCIA_FISICA_ENC","VIOLENCIA_PSICOLOGICA_ENC","VIOLENCIA_SEXUAL_ENC","VIOLENCIA_PATRIMONIAL_ENC"]].iloc[0].to_numpy().reshape(1, -1)
    X_eval_relation = input_data[['VIC_EST_CIV_ENC', 'VIC_REL_AGR_ENC', 'AGR_EST_CIV_ENC','AGR_TRABAJA_ENC','VIC_TRABAJA_ENC']].iloc[0].to_numpy().reshape(1, -1)

    probs_victim = models['victim'].predict_proba(X_eval_victim)
    probs_situation = models['situation'].predict_proba(X_eval_situation)
    probs_relation = models['relation'].predict_proba(X_eval_relation)

    # scores sin normalizar (ponderados por riesgo)
    raw_victim = score_from_probs(probs_victim, RIESGO_VICTIM)
    raw_situation = score_from_probs(probs_situation, RIESGO_SITUATION)
    raw_relation = score_from_probs(probs_relation, RIESGO_RELATION)

    # normalización (como en tu lógica original)
    prob_victim = raw_victim / len(RIESGO_VICTIM)
    prob_situation = raw_situation / len(RIESGO_SITUATION)
    prob_relation = raw_relation / len(RIESGO_RELATION)

    apoyo = 1.0 - i_apoyo_dict.get(depto, 0.0)
    combined_probs = (prob_relation + prob_situation + prob_victim + apoyo) / 4.0

    # resultado en formato legible
    result = {
        "IV_ES_PERFIL_VICTIM": float(prob_victim),
        "IV_ES_SITUATION": float(prob_situation),
        "IV_ES_RELATION": float(prob_relation),
        "IV_APOYO_INSTITUCIONAL_INV": float(apoyo),
        "IVESVI": float(combined_probs),
        "probs_victim": probs_victim.flatten().tolist(),
        "probs_situation": probs_situation.flatten().tolist(),
        "probs_relation": probs_relation.flatten().tolist()
    }
    return result

def render_results(res: Dict[str, Any]):
    st.subheader("Resultado resumido")
    col1, col2 = st.columns([1,1])
    with col1:
        st.metric("IVESVI (score)", format4(res["IVESVI"]))
        st.write("Composición (4 componentes, 4 decimales):")
        st.write(f"- Perfil víctima: {format4(res['IV_ES_PERFIL_VICTIM'])}")
        st.write(f"- Situación violencia: {format4(res['IV_ES_SITUATION'])}")
    with col2:
        st.write(f"- Relación víctima-agresor: {format4(res['IV_ES_RELATION'])}")
        st.write(f"- Apoyo institucional (inverso): {format4(res['IV_APOYO_INSTITUCIONAL_INV'])}")

    st.divider()
    st.subheader("Distribución de probabilidades por componente")
    # tablas con probabilidades y barras de progreso
    probs_v = pd.DataFrame({
        "Clase": [f"Clase {i}" for i in range(len(res['probs_victim']))],
        "Probabilidad": [round(p, 4) for p in res['probs_victim']]
    })
    probs_s = pd.DataFrame({
        "Clase": [f"Clase {i}" for i in range(len(res['probs_situation']))],
        "Probabilidad": [round(p, 4) for p in res['probs_situation']]
    })
    probs_r = pd.DataFrame({
        "Clase": [f"Clase {i}" for i in range(len(res['probs_relation']))],
        "Probabilidad": [round(p, 4) for p in res['probs_relation']]
    })

    st.caption("Probabilidades perfil víctima")
    st.table(probs_v)
    for i, p in enumerate(res['probs_victim']):
        st.progress(int(round(p * 100)), text=f"Perfil de la víctima {i}: {format4(p)}")

    st.caption("Probabilidades situación de violencia")
    st.table(probs_s)
    for i, p in enumerate(res['probs_situation']):
        st.progress(int(round(p * 100)), text=f"Perfil de la situación {i}: {format4(p)}")

    st.caption("Probabilidades relación víctima-agresor")
    st.table(probs_r)
    for i, p in enumerate(res['probs_relation']):
        st.progress(int(round(p * 100)), text=f"Perfil de la relación {i}: {format4(p)}")

    st.info("Interpretación: IVESVI cercano a 0 indica menor vulnerabilidad estructural; cercano a 1 indica mayor vulnerabilidad estructural. Valores mostrados con 4 decimales.")

# --- Interfaz -----------------------------------------------------------------
st.title('Índice de vulnerabilidad estructural sobre violencia intrafamiliar (IVESVI)')

label_encoders = load_label_encoders()
models = load_models()

departamento = st.selectbox(
    "Departamento donde se encuentra la víctima",
    ['guatemala', 'suchitepequez', 'ignorado', 'quetzaltenango',
     'huehuetenango', 'jalapa', 'escuintla', 'el progreso',
     'sacatepequez', 'chimaltenango', 'santa rosa', 'solola',
     'totonicapan', 'retalhuleu', 'san marcos', 'alta verapaz',
     'quiche', 'baja verapaz', 'peten', 'izabal', 'zacapa',
     'chiquimula', 'jutiapa']
)

st.header("Características de la víctima")
vic_alfab = st.selectbox("¿La víctima es alfabeta?", ["alfabeta", "analfabeta"])
vic_indigena = st.selectbox("¿La víctima pertenece a una etnia indígena?", ["si", "no"])
vic_escolaridad = st.selectbox("Nivel máximo de escolaridad de la víctima", [
    "basicos", "diversificado", "ninguno", "primaria", "secundaria", "universidad"
])
cantidad_hijos = st.selectbox("Cantidad de hijos", ["sin hijos", "hijo unico", "hijos medios", "muchos hijos"])

st.header("Características del agresor")
agr_alfab = st.selectbox("¿El agresor es alfabeto?", ["alfabeta", "analfabeta"])
agr_indigena = st.selectbox("¿El agresor pertenece a una etnia indígena?", ["si", "no"])
agr_escolaridad = st.selectbox("Nivel máximo de escolaridad del agresor", [
    "basicos", "diversificado", "ninguno", "primaria", "secundaria", "universidad"
])
agr_sexo = st.selectbox("Sexo del agresor", ["hombres", "mujeres"])

st.header("Tipos de violencia ejercida")
violencia_fisica = st.selectbox("¿Ha habido violencia física?", ["presente", "no presente"])
violencia_psicologica = st.selectbox("¿Ha habido violencia psicológica?", ["presente", "no presente"])
violencia_sexual = st.selectbox("¿Ha habido violencia sexual?", ["presente", "no presente"])
violencia_patrimonial = st.selectbox("¿Ha habido violencia patrimonial?", ["presente", "no presente"])

st.header("Dinámica relacional")
vic_est_civ = st.selectbox("Estado civil de la víctima", ["casados(as)", "otro", "solteros(as)", "unidos(as)", "viudos(as)"])
vic_rel_agr = st.selectbox("Relación de la víctima con el agresor", [
    "conviviente", "esposos(as)", "ex-conyuges", "hermanos(as)", "hijastros(as)", "hijos(as)", "nietos(as)", "otro pariente", "padres/madres", "suegros(as)"
])
agr_est_civ = st.selectbox("Estado civil del agresor", ["casados(as)", "otro", "solteros(as)", "unidos(as)", "viudos(as)"])
agr_trabaja2 = st.selectbox("¿El agresor trabaja?", ["si", "no"])
vic_trabaja2 = st.selectbox("¿La víctima trabaja?", ["si", "no"])

# --- Action ---------------------------------------------------------------
if st.button("Calcular IVESVI"):
    # montar un diccionario crudo
    raw_input = {
        "VIC_ALFAB": vic_alfab,
        "VIC_ES_INDIGENA": vic_indigena,
        "VIC_NIV_ESCOLARIDAD": vic_escolaridad,
        "CANTIDAD_HIJOS": cantidad_hijos,
        "AGR_ALFAB": agr_alfab,
        "AGR_ES_INDIGENA": agr_indigena,
        "AGR_NIV_ESCOLARIDAD": agr_escolaridad,
        "AGR_SEXO": agr_sexo,
        "VIOLENCIA_FISICA": violencia_fisica,
        "VIOLENCIA_PSICOLOGICA": violencia_psicologica,
        "VIOLENCIA_SEXUAL": violencia_sexual,
        "VIOLENCIA_PATRIMONIAL": violencia_patrimonial,
        "VIC_EST_CIV": vic_est_civ,
        "VIC_REL_AGR": vic_rel_agr,
        "AGR_EST_CIV": agr_est_civ,
        "AGR_TRABAJA": agr_trabaja2,
        "VIC_TRABAJA": vic_trabaja2
    }

    try:
        input_df = prepare_input_dataframe(raw_input, label_encoders)
    except Exception as e:
        st.error("Error preparando los datos: " + str(e))
    else:
        if not models:
            st.error("Modelos no disponibles. Revisa la consola o los archivos en /models.")
        else:
            with st.spinner("Calculando IVESVI..."):
                res = gmm_full_predict(input_df, departamento, models)
            render_results(res)
            st.success("Cálculo finalizado")
