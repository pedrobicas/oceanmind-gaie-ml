import csv
import math
import random
from pathlib import Path

OUTPUT = Path("data/oceanmind_dataset.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

REGIONS = [
    ("Atlântico Sul", -30, -10, -55, 10),
    ("Costa Nordeste", -10, 2, -45, -30),
    ("Pacífico Equatorial", -5, 5, -150, -90),
    ("Índico Oeste", -25, 5, 40, 80),
    ("Atlântico Norte", 10, 40, -70, -10),
    ("Pacífico Sul", -40, -10, -170, -80),
]

def classify_risk(score: float) -> str:
    if score < 1.5:
        return "baixo"
    if score < 3.5:
        return "moderado"
    if score < 5.8:
        return "alto"
    return "critico"

def main(rows: int = 1600, seed: int = 42) -> None:
    random.seed(seed)
    data = []

    for i in range(rows):
        region, lat_min, lat_max, lon_min, lon_max = random.choice(REGIONS)
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        month = random.randint(1, 12)

        seasonal = math.sin((month - 1) / 12 * 2 * math.pi)
        base_temp = 26 + (5 if abs(lat) < 10 else 2 if abs(lat) < 25 else -1) + 1.2 * seasonal

        sst = random.gauss(base_temp, 1.2)
        anomaly = max(-1.5, random.gauss(0.6, 0.9))
        if random.random() < 0.18:
            anomaly += random.uniform(1.0, 2.4)

        wind = max(0, random.gauss(22, 10) + anomaly * 3)
        wave = max(0.2, random.gauss(1.7, 0.7) + wind / 35 + anomaly * 0.25)
        humidity = min(100, max(40, random.gauss(76, 10)))
        pressure = max(970, min(1035, random.gauss(1012, 9) - wind * 0.15 - anomaly * 2))
        cloud = min(100, max(0, random.gauss(45, 22) + humidity * 0.15 + anomaly * 4))
        chlorophyll = max(0.02, random.gauss(0.8, 0.4) + max(0, anomaly) * 0.12)
        current_speed = max(0.05, random.gauss(0.45, 0.25) + wave * 0.05)

        score = 0
        score += anomaly * 2.2
        score += max(0, sst - 28) * 0.7
        score += max(0, wind - 28) * 0.05
        score += max(0, wave - 2.5) * 0.7
        score += max(0, 1005 - pressure) * 0.08
        score += max(0, cloud - 65) * 0.02
        score += max(0, chlorophyll - 1.2) * 0.6
        score += random.gauss(0, 0.6)

        data.append({
            "sample_id": f"OM-{i+1:04d}",
            "region": region,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "month": month,
            "sea_surface_temperature_c": round(sst, 2),
            "temperature_anomaly_c": round(anomaly, 2),
            "wind_speed_kmh": round(wind, 2),
            "wave_height_m": round(wave, 2),
            "humidity_percent": round(humidity, 2),
            "pressure_hpa": round(pressure, 2),
            "cloud_coverage_percent": round(cloud, 2),
            "chlorophyll_index": round(chlorophyll, 3),
            "current_speed_ms": round(current_speed, 3),
            "ocean_risk_level": classify_risk(score),
        })

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)

    print(f"Dataset gerado em {OUTPUT} com {len(data)} linhas.")

if __name__ == "__main__":
    main()
