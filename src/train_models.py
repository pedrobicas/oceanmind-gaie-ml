from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/oceanmind_dataset.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "ocean_risk_level"
DROP_COLUMNS = ["sample_id", TARGET]

def build_pipeline(model):
    numeric_features = [
        "latitude",
        "longitude",
        "month",
        "sea_surface_temperature_c",
        "temperature_anomaly_c",
        "wind_speed_kmh",
        "wave_height_m",
        "humidity_percent",
        "pressure_hpa",
        "cloud_coverage_percent",
        "chlorophyll_index",
        "current_speed_ms",
    ]
    categorical_features = ["region"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=220,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42,
            learning_rate=0.08,
            n_estimators=180,
            max_depth=3,
        ),
    }

    results = {}

    for name, model in candidates.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

        results[name] = {
            "accuracy": accuracy,
            "classification_report": report,
            "labels": list(pipeline.classes_),
        }

        joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")

        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=pipeline.classes_)
        disp.plot(xticks_rotation=45)
        plt.title(f"Matriz de confusão — {name}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"confusion_matrix_{name}.png")
        plt.close()

    best_name = max(results, key=lambda item: results[item]["accuracy"])
    best_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(joblib.load(MODELS_DIR / f"{best_name}.joblib"), best_path)

    summary = {
        "best_model": best_name,
        "best_accuracy": results[best_name]["accuracy"],
        "all_results": results,
    }

    with (REPORTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Treinamento finalizado.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
