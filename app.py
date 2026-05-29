import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/best_model.joblib")
METRICS_PATH = Path("reports/metrics.json")
SHAP_RANKING_PATH = Path("reports/shap_ranking.json")
SHAP_FIGURE_PATH = Path("reports/figures/shap_feature_importance.png")

FEATURE_LABELS = {
    "num__temperature_anomaly_c": "Anomalia térmica",
    "num__sea_surface_temperature_c": "Temperatura do mar",
    "num__wave_height_m": "Altura das ondas",
    "num__wind_speed_kmh": "Velocidade do vento",
    "num__pressure_hpa": "Pressão atmosférica",
    "num__latitude": "Latitude",
    "num__cloud_coverage_percent": "Cobertura de nuvens",
    "num__chlorophyll_index": "Índice de clorofila",
    "num__current_speed_ms": "Velocidade da corrente",
    "num__month": "Mês",
}

PRESETS = {
    "Condição estável": {
        "region": "Atlântico Sul",
        "latitude": -23.5,
        "longitude": -42.1,
        "month": 6,
        "sea_surface_temperature_c": 25.8,
        "temperature_anomaly_c": 0.2,
        "wind_speed_kmh": 18.0,
        "wave_height_m": 1.4,
        "humidity_percent": 72.0,
        "pressure_hpa": 1015.0,
        "cloud_coverage_percent": 35.0,
        "chlorophyll_index": 0.72,
        "current_speed_ms": 0.42,
    },
    "Atenção costeira": {
        "region": "Costa Nordeste",
        "latitude": -8.4,
        "longitude": -34.9,
        "month": 9,
        "sea_surface_temperature_c": 28.6,
        "temperature_anomaly_c": 1.1,
        "wind_speed_kmh": 31.0,
        "wave_height_m": 2.3,
        "humidity_percent": 81.0,
        "pressure_hpa": 1008.0,
        "cloud_coverage_percent": 62.0,
        "chlorophyll_index": 1.08,
        "current_speed_ms": 0.62,
    },
    "Alerta severo": {
        "region": "Atlântico Sul",
        "latitude": -23.5,
        "longitude": -42.1,
        "month": 6,
        "sea_surface_temperature_c": 29.4,
        "temperature_anomaly_c": 2.1,
        "wind_speed_kmh": 42.0,
        "wave_height_m": 3.1,
        "humidity_percent": 83.0,
        "pressure_hpa": 1002.0,
        "cloud_coverage_percent": 76.0,
        "chlorophyll_index": 1.15,
        "current_speed_ms": 0.72,
    },
}

RISK_TEXT = {
    "baixo": {
        "label": "Baixo",
        "class": "risk-low",
        "summary": "Condição estável para monitoramento de rotina.",
        "action": "Manter acompanhamento periódico dos indicadores ambientais.",
    },
    "moderado": {
        "label": "Moderado",
        "class": "risk-moderate",
        "summary": "Há sinais de atenção, mas sem indicação crítica no momento.",
        "action": "Acompanhar tendência de temperatura, vento, ondas e pressão.",
    },
    "alto": {
        "label": "Alto",
        "class": "risk-high",
        "summary": "Condição relevante de risco oceânico/climático.",
        "action": "Priorizar análise da região e verificar evolução nas próximas medições.",
    },
    "critico": {
        "label": "Crítico",
        "class": "risk-critical",
        "summary": "Sinais fortes de risco ambiental ou climático.",
        "action": "Acionar plano de alerta e revisar dados operacionais da região.",
    },
}


st.set_page_config(
    page_title="OceanMind IA",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }
    .hero {
        border: 1px solid #d7e3ea;
        border-radius: 8px;
        padding: 1.35rem 1.5rem;
        background: linear-gradient(135deg, #f7fbfc 0%, #eef7f8 100%);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0 0 .35rem 0;
        font-size: 2rem;
        color: #12323d;
    }
    .hero p {
        margin: 0;
        color: #425b64;
        font-size: 1.02rem;
    }
    .hint-box {
        border: 1px solid #d8e2e7;
        border-radius: 8px;
        padding: .9rem 1rem;
        background: #ffffff;
        color: #344b54;
        margin-bottom: 1rem;
    }
    .result-card {
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        color: #0f252d;
        border: 1px solid transparent;
        min-height: 165px;
    }
    .risk-low {
        background: #edf8f0;
        border-color: #b9e2c4;
    }
    .risk-moderate {
        background: #fff8e6;
        border-color: #edd48c;
    }
    .risk-high {
        background: #fff1e8;
        border-color: #f1b98d;
    }
    .risk-critical {
        background: #fdecec;
        border-color: #eca2a2;
    }
    .result-label {
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .04em;
        color: #536a72;
        margin-bottom: .35rem;
    }
    .result-value {
        font-size: 2.15rem;
        font-weight: 750;
        margin: 0 0 .4rem 0;
    }
    .result-copy {
        margin: .25rem 0;
        color: #324950;
    }
    .metric-card {
        border: 1px solid #dbe5e9;
        border-radius: 8px;
        padding: .8rem 1rem;
        background: #ffffff;
        min-height: 94px;
    }
    .metric-card strong {
        display: block;
        font-size: 1.35rem;
        color: #143541;
    }
    .metric-card span {
        color: #627780;
        font-size: .92rem;
    }
    div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 700;
        min-height: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_shap_ranking():
    if not SHAP_RANKING_PATH.exists():
        return None
    ranking = json.loads(SHAP_RANKING_PATH.read_text(encoding="utf-8"))
    rows = []
    for index, item in enumerate(ranking, start=1):
        feature = item["feature"]
        rows.append(
            {
                "posição": index,
                "variável": FEATURE_LABELS.get(feature, feature.replace("num__", "").replace("cat__", "")),
                "importância": round(float(item["importance"]), 4),
            }
        )
    return pd.DataFrame(rows)


def build_metrics_table(metrics):
    rows = []
    for model_name, result in metrics["all_results"].items():
        report = result["classification_report"]["weighted avg"]
        rows.append(
            {
                "modelo": "Random Forest" if model_name == "random_forest" else "Gradient Boosting",
                "acurácia": f"{result['accuracy']:.2%}",
                "precision": f"{report['precision']:.2%}",
                "recall": f"{report['recall']:.2%}",
                "f1-score": f"{report['f1-score']:.2%}",
            }
        )
    return pd.DataFrame(rows)


def render_result(prediction, confidence):
    risk = RISK_TEXT[prediction]
    confidence_text = f"{confidence:.1%}" if confidence is not None else "Não disponível"
    st.markdown(
        f"""
        <div class="result-card {risk['class']}">
            <div class="result-label">Nível de risco previsto</div>
            <div class="result-value">{risk['label']}</div>
            <p class="result-copy">{risk['summary']}</p>
            <p class="result-copy"><strong>Ação sugerida:</strong> {risk['action']}</p>
            <p class="result-copy"><strong>Confiança do modelo:</strong> {confidence_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_summary(input_data):
    labels = {
        "region": "Região",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "month": "Mês",
        "sea_surface_temperature_c": "Temperatura do mar (°C)",
        "temperature_anomaly_c": "Anomalia térmica (°C)",
        "wind_speed_kmh": "Vento (km/h)",
        "wave_height_m": "Ondas (m)",
        "humidity_percent": "Umidade (%)",
        "pressure_hpa": "Pressão (hPa)",
        "cloud_coverage_percent": "Nuvens (%)",
        "chlorophyll_index": "Clorofila",
        "current_speed_ms": "Corrente (m/s)",
    }
    summary = [{"indicador": labels[key], "valor": value} for key, value in input_data.items()]
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


if not MODEL_PATH.exists():
    st.warning("Modelo não encontrado. Rode primeiro: `python src/train_models.py`.")
    st.stop()

model = load_model()
metrics = load_metrics()
shap_ranking = load_shap_ranking()

st.markdown(
    """
    <div class="hero">
        <h1>🌊 OceanMind IA</h1>
        <p>Simule uma leitura ambiental e veja o nível de risco oceânico previsto pelo modelo.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hint-box">
        <strong>Como usar:</strong> escolha um cenário pronto ou ajuste os indicadores.
        Depois clique em <strong>Classificar risco</strong>. Valores altos de anomalia térmica,
        ondas, vento e queda de pressão tendem a aumentar o risco previsto.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Cenário")
    preset_name = st.selectbox(
        "Comece por uma situação típica",
        list(PRESETS.keys()),
        index=2,
        help="Os cenários servem como ponto de partida. Você pode ajustar qualquer valor depois.",
    )
    preset = PRESETS[preset_name]

    st.caption("O modelo foi treinado com dados sintéticos de regiões oceânicas e climáticas.")

    if metrics:
        st.markdown("---")
        st.metric("Melhor modelo", "Random Forest")
        st.metric("Acurácia de teste", f"{metrics['best_accuracy']:.2%}")

st.subheader("1. Dados da região")
region_col, month_col, lat_col, lon_col = st.columns([1.4, 0.8, 1, 1])

with region_col:
    region = st.selectbox(
        "Região monitorada",
        ["Atlântico Sul", "Costa Nordeste", "Pacífico Equatorial", "Índico Oeste", "Atlântico Norte", "Pacífico Sul"],
        index=["Atlântico Sul", "Costa Nordeste", "Pacífico Equatorial", "Índico Oeste", "Atlântico Norte", "Pacífico Sul"].index(preset["region"]),
    )

with month_col:
    month = st.slider("Mês", 1, 12, int(preset["month"]))

with lat_col:
    latitude = st.number_input("Latitude", value=float(preset["latitude"]), format="%.4f")

with lon_col:
    longitude = st.number_input("Longitude", value=float(preset["longitude"]), format="%.4f")

st.subheader("2. Indicadores ambientais")
ocean_col, weather_col = st.columns(2)

with ocean_col:
    st.markdown("**Oceano**")
    sea_surface_temperature_c = st.slider(
        "Temperatura da superfície do mar (°C)",
        18.0,
        34.0,
        float(preset["sea_surface_temperature_c"]),
        help="Temperaturas mais altas podem indicar aquecimento local.",
    )
    temperature_anomaly_c = st.slider(
        "Anomalia térmica (°C)",
        -2.0,
        5.0,
        float(preset["temperature_anomaly_c"]),
        help="Diferença em relação ao padrão esperado para a região e época.",
    )
    wave_height_m = st.slider(
        "Altura das ondas (m)",
        0.0,
        8.0,
        float(preset["wave_height_m"]),
        help="Ondas mais altas podem indicar instabilidade marítima.",
    )
    current_speed_ms = st.slider(
        "Velocidade da corrente (m/s)",
        0.0,
        2.0,
        float(preset["current_speed_ms"]),
    )
    chlorophyll_index = st.slider(
        "Índice de clorofila",
        0.0,
        3.0,
        float(preset["chlorophyll_index"]),
        help="Ajuda a representar alterações biológicas ou produtividade oceânica.",
    )

with weather_col:
    st.markdown("**Clima**")
    wind_speed_kmh = st.slider(
        "Velocidade do vento (km/h)",
        0.0,
        90.0,
        float(preset["wind_speed_kmh"]),
    )
    pressure_hpa = st.slider(
        "Pressão atmosférica (hPa)",
        960.0,
        1040.0,
        float(preset["pressure_hpa"]),
        help="Pressão mais baixa pode estar associada a sistemas meteorológicos instáveis.",
    )
    humidity_percent = st.slider("Umidade (%)", 30.0, 100.0, float(preset["humidity_percent"]))
    cloud_coverage_percent = st.slider(
        "Cobertura de nuvens (%)",
        0.0,
        100.0,
        float(preset["cloud_coverage_percent"]),
    )

input_data = {
    "region": region,
    "latitude": latitude,
    "longitude": longitude,
    "month": month,
    "sea_surface_temperature_c": sea_surface_temperature_c,
    "temperature_anomaly_c": temperature_anomaly_c,
    "wind_speed_kmh": wind_speed_kmh,
    "wave_height_m": wave_height_m,
    "humidity_percent": humidity_percent,
    "pressure_hpa": pressure_hpa,
    "cloud_coverage_percent": cloud_coverage_percent,
    "chlorophyll_index": chlorophyll_index,
    "current_speed_ms": current_speed_ms,
}

st.subheader("3. Resultado")

if st.button("Classificar risco", type="primary"):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    confidence = None

    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(df).max())

    result_col, summary_col = st.columns([1, 1.25])

    with result_col:
        render_result(prediction, confidence)

    with summary_col:
        st.markdown("**Dados usados na classificação**")
        render_input_summary(input_data)
else:
    st.info("Ajuste os indicadores e clique em **Classificar risco** para ver a previsão.")

st.divider()

st.subheader("Entenda o modelo")

metric_col, shap_col = st.columns([1, 1])

with metric_col:
    st.markdown("**Desempenho no teste**")
    if metrics:
        top_a, top_b = st.columns(2)
        with top_a:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span>Melhor modelo</span>
                    <strong>Random Forest</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with top_b:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span>Acurácia</span>
                    <strong>{metrics['best_accuracy']:.2%}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.dataframe(build_metrics_table(metrics), use_container_width=True, hide_index=True)
    else:
        st.info("Rode o treinamento para gerar `reports/metrics.json`.")

with shap_col:
    st.markdown("**Variáveis que mais pesam na decisão**")
    if shap_ranking is not None:
        st.dataframe(shap_ranking.head(6), use_container_width=True, hide_index=True)
        if SHAP_FIGURE_PATH.exists():
            with st.expander("Ver gráfico SHAP"):
                st.image(str(SHAP_FIGURE_PATH), use_column_width=True)
    else:
        st.info("Rode `python src/shap_analysis.py` para gerar o ranking SHAP.")

