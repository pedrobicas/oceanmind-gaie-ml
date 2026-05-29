from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("models/best_model.joblib")

def predict(input_data: dict) -> dict:
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([input_data])
    pred = model.predict(df)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df)[0]
        confidence = float(probabilities.max())

    return {
        "risk_level": pred,
        "confidence": confidence,
    }

if __name__ == "__main__":
    sample = {
        "region": "Atlântico Sul",
        "latitude": -23.5,
        "longitude": -42.1,
        "month": 6,
        "sea_surface_temperature_c": 29.4,
        "temperature_anomaly_c": 2.1,
        "wind_speed_kmh": 42,
        "wave_height_m": 3.1,
        "humidity_percent": 83,
        "pressure_hpa": 1002,
        "cloud_coverage_percent": 76,
        "chlorophyll_index": 1.15,
        "current_speed_ms": 0.72,
    }
    print(predict(sample))
