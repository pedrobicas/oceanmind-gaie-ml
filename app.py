import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/best_model.joblib")
METRICS_PATH = Path("reports/metrics.json")
SHAP_RANKING_PATH = Path("reports/shap_ranking.json")

st.set_page_config(
    page_title="OceanMind IA",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 OceanMind — Previsão de Risco Oceânico")
st.write(
    "Aplicação demonstrativa de Machine Learning para classificar risco oceânico/climático "
    "a partir de dados espaciais, climáticos e ambientais."
)

if not MODEL_PATH.exists():
    st.warning("Modelo não encontrado. Rode primeiro: `python src/train_models.py`")
    st.stop()

model = joblib.load(MODEL_PATH)

col1, col2, col3 = st.columns(3)

with col1:
    region = st.selectbox(
        "Região",
        ["Atlântico Sul", "Costa Nordeste", "Pacífico Equatorial", "Índico Oeste", "Atlântico Norte", "Pacífico Sul"],
    )
    latitude = st.number_input("Latitude", value=-23.5)
    longitude = st.number_input("Longitude", value=-42.1)
    month = st.slider("Mês", 1, 12, 6)

with col2:
    sea_surface_temperature_c = st.slider("Temperatura da superfície do mar (°C)", 18.0, 34.0, 29.4)
    temperature_anomaly_c = st.slider("Anomalia térmica (°C)", -2.0, 5.0, 2.1)
    wind_speed_kmh = st.slider("Vento (km/h)", 0.0, 90.0, 42.0)
    wave_height_m = st.slider("Altura das ondas (m)", 0.0, 8.0, 3.1)

with col3:
    humidity_percent = st.slider("Umidade (%)", 30.0, 100.0, 83.0)
    pressure_hpa = st.slider("Pressão atmosférica (hPa)", 960.0, 1040.0, 1002.0)
    cloud_coverage_percent = st.slider("Cobertura de nuvens (%)", 0.0, 100.0, 76.0)
    chlorophyll_index = st.slider("Índice de clorofila", 0.0, 3.0, 1.15)
    current_speed_ms = st.slider("Velocidade da corrente (m/s)", 0.0, 2.0, 0.72)

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

if st.button("Classificar risco"):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(df).max())

    st.subheader("Resultado")
    st.metric("Nível de risco previsto", prediction.upper())

    if confidence is not None:
        st.metric("Confiança aproximada", f"{confidence:.1%}")

    st.json(input_data)

st.divider()

col_metrics, col_shap = st.columns(2)

with col_metrics:
    st.subheader("Métricas do treinamento")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.write(f"Melhor modelo: **{metrics['best_model']}**")
        st.write(f"Acurácia: **{metrics['best_accuracy']:.2%}**")
        st.json(metrics["all_results"])
    else:
        st.info("Rode o treinamento para gerar `reports/metrics.json`.")

with col_shap:
    st.subheader("Interpretabilidade com SHAP")
    if SHAP_RANKING_PATH.exists():
        ranking = json.loads(SHAP_RANKING_PATH.read_text(encoding="utf-8"))
        st.table(pd.DataFrame(ranking))
    else:
        st.info("Rode `python src/shap_analysis.py` para gerar o ranking SHAP.")
