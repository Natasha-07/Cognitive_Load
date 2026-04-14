from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

SEED = 42
TEST_SIZE = 0.20
BACKGROUND_SAMPLES = 200
EXPLAIN_SAMPLES = 500

EEG_BAND_LABELS = {
    "EEG_0": "Delta (<4 Hz)",
    "EEG_1": "Theta (4-8 Hz)",
    "EEG_2": "Alpha (8-12 Hz)",
    "EEG_3": "Beta (13-30 Hz)",
}


def locate_synthetic_dataset() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "synthetic_mental_health_data.csv",
        script_dir.parent / "synthetic_mental_health_data.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find synthetic_mental_health_data.csv")


def load_notebook_style_synthetic_data(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path).copy()
    df["Cognitive_State"] = df["Cognitive_State"].astype(str).str.strip()

    label_map = {
        "anxious": 1,
        "stressed": 1,
        "overloaded": 1,
        "distracted": 1,
        "Anxious": 1,
        "Stressed": 1,
        "Overloaded": 1,
        "Distracted": 1,
        "Cognitive Overload": 1,
        "calm": 0,
        "focused": 0,
        "Calm": 0,
        "Focused": 0,
    }

    df["y_bin"] = df["Cognitive_State"].map(label_map)
    df = df.dropna(subset=["y_bin"]).reset_index(drop=True)
    df["y_bin"] = df["y_bin"].astype(int)

    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    base_cols = [col for col in ["GSR_Values", "Age", "Duration (minutes)"] if col in df.columns]
    eeg_cols = [col for col in df.columns if col.startswith("EEG_")]
    prep_cols = [col for col in df.columns if col.startswith("Preprocessed_")]
    demo_cols = [col for col in ["Gender"] if col in df.columns]

    feature_cols = base_cols + eeg_cols + prep_cols + demo_cols
    X = df[feature_cols].copy()

    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    y = df["y_bin"].to_numpy(dtype=int)
    return X, y


def build_mlp_model() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=SEED,
    )


def select_positive_class(shap_values: shap.Explanation) -> shap.Explanation:
    if shap_values.values.ndim == 3:
        base_values = np.asarray(shap_values.base_values)
        if base_values.ndim == 2:
            base_values = base_values[:, 1]
        return shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=base_values,
            data=shap_values.data,
            feature_names=shap_values.feature_names,
        )
    return shap_values


def compute_metrics(model: MLPClassifier, X_test: pd.DataFrame, y_test: np.ndarray) -> pd.DataFrame:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return pd.DataFrame(
        [
            {
                "Model": "MLP",
                "Accuracy": float(accuracy_score(y_test, y_pred)),
                "Balanced_Accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "F1": float(f1_score(y_test, y_pred, zero_division=0)),
                "ROC_AUC": float(roc_auc_score(y_test, y_prob)),
                "Test_Size": int(len(y_test)),
            }
        ]
    )


def plot_global_importance(importance_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = importance_df.sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#1E88E5")
    ax.set_title("SHAP Global Feature Importance (Threat Class)\n(MLP Model)", fontsize=16, weight="bold")
    ax.set_xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_eeg_band_importance(eeg_df: pd.DataFrame, output_path: Path) -> None:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(eeg_df["band"], eeg_df["mean_abs_shap"], color=colors[: len(eeg_df)])
    ax.set_title("EEG Frequency Band Importance for Threat Detection\n(MLP Model)", fontsize=18, weight="bold")
    ax.set_xlabel("EEG Frequency Band")
    ax.set_ylabel("Mean Absolute SHAP Value")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)

    for bar, value in zip(bars, eeg_df["mean_abs_shap"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(SEED)

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = locate_synthetic_dataset()
    X, y = load_notebook_style_synthetic_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns,
        index=X_test.index,
    )

    model = build_mlp_model()
    model.fit(X_train_scaled, y_train)

    metrics_df = compute_metrics(model, X_test_scaled, y_test)

    background = X_train_scaled.sample(n=min(BACKGROUND_SAMPLES, len(X_train_scaled)), random_state=SEED)
    explain_df = X_test_scaled.sample(n=min(EXPLAIN_SAMPLES, len(X_test_scaled)), random_state=SEED)

    explainer = shap.Explainer(
        model.predict_proba,
        background,
        algorithm="permutation",
        feature_names=list(X_train_scaled.columns),
    )
    shap_values = explainer(explain_df, max_evals=(2 * explain_df.shape[1]) + 1)
    shap_values_pos = select_positive_class(shap_values)

    importance_df = (
        pd.DataFrame(
            {
                "feature": list(shap_values_pos.feature_names),
                "mean_abs_shap": np.abs(shap_values_pos.values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    )

    eeg_rows: list[dict[str, str | float]] = []
    for eeg_feature, band_label in EEG_BAND_LABELS.items():
        value = float(
            importance_df.loc[importance_df["feature"] == eeg_feature, "mean_abs_shap"].iloc[0]
        )
        eeg_rows.append(
            {
                "feature": eeg_feature,
                "band": band_label,
                "mean_abs_shap": value,
            }
        )

    eeg_df = pd.DataFrame(eeg_rows).sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    importance_csv = output_dir / "mlp_shap_feature_importance.csv"
    eeg_csv = output_dir / "mlp_eeg_band_importance.csv"
    metrics_csv = output_dir / "mlp_synthetic_test_metrics.csv"
    global_png = output_dir / "mlp_shap_global_feature_importance_threat_class.png"
    eeg_png = output_dir / "mlp_eeg_frequency_band_importance_threat_detection.png"

    importance_df.to_csv(importance_csv, index=False)
    eeg_df.to_csv(eeg_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)

    plot_global_importance(importance_df, global_png)
    plot_eeg_band_importance(eeg_df, eeg_png)

    print("Synthetic dataset:", csv_path)
    print("Saved files:")
    print(f"- {importance_csv}")
    print(f"- {eeg_csv}")
    print(f"- {metrics_csv}")
    print(f"- {global_png}")
    print(f"- {eeg_png}")
    print("\nMLP synthetic holdout metrics:")
    print(metrics_df.to_string(index=False))
    print("\nTop SHAP features:")
    print(importance_df.head(10).to_string(index=False))
    print("\nEEG band importance:")
    print(eeg_df.to_string(index=False))


if __name__ == "__main__":
    main()
