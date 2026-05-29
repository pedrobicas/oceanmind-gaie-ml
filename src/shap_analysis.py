from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/oceanmind_dataset.csv")
MODEL_PATH = Path("models/best_model.joblib")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "ocean_risk_level"
DROP_COLUMNS = ["sample_id", TARGET]

def aggregate_importance(shap_values, n_features: int):
    values = getattr(shap_values, "values", shap_values)
    values = np.asarray(values)

    if values.ndim == 2:
        return np.abs(values).mean(axis=0)

    if values.ndim == 3:
        feature_axes = [axis for axis, size in enumerate(values.shape) if size == n_features]
        if not feature_axes:
            raise ValueError(f"Não foi possível identificar o eixo de features em SHAP: {values.shape}")

        feature_axis = feature_axes[-1]
        aggregate_axes = tuple(axis for axis in range(values.ndim) if axis != feature_axis)
        return np.abs(values).mean(axis=aggregate_axes)

    raise ValueError(f"Formato de SHAP não suportado: {values.shape}")

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import shap
    except ImportError:
        raise SystemExit("Instale o SHAP com: pip install shap")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=DROP_COLUMNS)
    pipeline = joblib.load(MODEL_PATH)

    sample = X.sample(min(250, len(X)), random_state=42)
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    # SHAP para modelos de árvore. Em alguns modelos multi-classe o retorno pode ser uma lista.
    explainer = shap.Explainer(model, X_transformed)
    shap_values = explainer(X_transformed, check_additivity=False)

    importance = aggregate_importance(shap_values, len(feature_names))

    ranking = sorted(
        zip(feature_names, importance),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    plot_features = [str(feature) for feature, _ in reversed(ranking)]
    plot_values = [float(value) for _, value in reversed(ranking)]

    plt.figure(figsize=(10, 6))
    plt.barh(plot_features, plot_values)
    plt.xlabel("Importância média absoluta SHAP")
    plt.title("Principais variáveis por importância SHAP")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_feature_importance.png")
    plt.close()

    with (REPORTS_DIR / "shap_ranking.json").open("w", encoding="utf-8") as f:
        json.dump(
            [{"feature": str(feature), "importance": float(value)} for feature, value in ranking],
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Análise SHAP gerada em reports/figures/shap_feature_importance.png")
    print("Ranking salvo em reports/shap_ranking.json")

if __name__ == "__main__":
    main()
