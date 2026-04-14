from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 42
N_FOLDS = 5
FINAL_TEST_SIZE = 0.20
N_JOBS = 1


class Torch1DCNNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        epochs: int = 8,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        dropout: float = 0.3,
        patience: int = 2,
        random_state: int = SEED,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.patience = patience
        self.random_state = random_state
        self.model_: nn.Module | None = None
        self.input_length_: int | None = None
        self.classes_: np.ndarray | None = None

    def _set_seed(self) -> None:
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    def _to_numpy(self, X: Any) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()
        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {X_np.shape}")
        return X_np

    def _build_model(self, input_length: int) -> nn.Module:
        pooled_len = min(4, input_length)
        return nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(pooled_len),
            nn.Flatten(),
            nn.Linear(16 * pooled_len, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1),
        )

    def fit(self, X: Any, y: Any) -> "Torch1DCNNClassifier":
        self._set_seed()

        X_np = self._to_numpy(X)
        y_np = np.asarray(y, dtype=np.int64)
        self.input_length_ = X_np.shape[1]
        self.classes_ = np.unique(y_np)

        X_train, X_val, y_train, y_val = train_test_split(
            X_np,
            y_np,
            test_size=0.15,
            random_state=self.random_state,
            stratify=y_np,
        )

        train_x_tensor = torch.from_numpy(X_train).unsqueeze(1)
        train_y_tensor = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
        val_x_tensor = torch.from_numpy(X_val).unsqueeze(1)
        val_y_tensor = torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1)

        train_ds = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model_ = self._build_model(self.input_length_)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        pos_count = float((y_train == 1).sum())
        neg_count = float((y_train == 0).sum())
        pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_state = None
        best_val_loss = float("inf")
        no_improve = 0

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(val_x_tensor)
                val_loss = float(criterion(val_logits, val_y_tensor).item())
            self.model_.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        X_np = self._to_numpy(X)
        x_tensor = torch.from_numpy(X_np).unsqueeze(1)
        with torch.no_grad():
            probs_pos = torch.sigmoid(self.model_(x_tensor)).cpu().numpy().reshape(-1)
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def parse_vector_column(
    series: pd.Series,
    expected_length: int,
    prefix: str,
) -> pd.DataFrame:
    rows: list[list[float]] = []

    for value in series:
        parsed: list[float]
        if isinstance(value, (list, tuple, np.ndarray)):
            parsed = list(value)
        elif isinstance(value, str):
            try:
                parsed = list(ast.literal_eval(value))
            except (ValueError, SyntaxError):
                parsed = []
        else:
            parsed = []

        parsed = parsed[:expected_length]
        if len(parsed) < expected_length:
            parsed = parsed + [np.nan] * (expected_length - len(parsed))

        numeric_row = []
        for item in parsed:
            try:
                numeric_row.append(float(item))
            except (TypeError, ValueError):
                numeric_row.append(np.nan)
        rows.append(numeric_row)

    columns = [f"{prefix}_{idx}" for idx in range(expected_length)]
    return pd.DataFrame(rows, columns=columns, index=series.index)


def prepare_dataset(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path).copy()

    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    if "EEG_0" not in df.columns and "EEG_Frequency_Bands" in df.columns:
        eeg_df = parse_vector_column(df["EEG_Frequency_Bands"], expected_length=4, prefix="EEG")
        df = pd.concat([df.drop(columns=["EEG_Frequency_Bands"]), eeg_df], axis=1)

    if "Preprocessed_0" not in df.columns and "Preprocessed_Features" in df.columns:
        prep_df = parse_vector_column(df["Preprocessed_Features"], expected_length=2, prefix="Preprocessed")
        df = pd.concat([df.drop(columns=["Preprocessed_Features"]), prep_df], axis=1)

    state_series = df["Cognitive_State"].astype(str).str.strip().str.lower()
    label_map = {
        "anxious": 1,
        "stressed": 1,
        "overloaded": 1,
        "distracted": 1,
        "cognitive overload": 1,
        "calm": 0,
        "focused": 0,
    }
    y_series = state_series.map(label_map)
    keep_mask = y_series.notna()
    df = df.loc[keep_mask].reset_index(drop=True)
    y = y_series.loc[keep_mask].astype(int).to_numpy()

    numeric_candidates = ["GSR_Values", "Age", "Duration (minutes)"]
    numeric_candidates += sorted([c for c in df.columns if c.startswith("EEG_")])
    numeric_candidates += sorted([c for c in df.columns if c.startswith("Preprocessed_")])

    categorical_candidates = ["Gender", "Session_Type", "Environmental_Context"]

    numeric_features = [c for c in numeric_candidates if c in df.columns]
    categorical_features = [c for c in categorical_candidates if c in df.columns]
    feature_cols = numeric_features + categorical_features

    if not feature_cols:
        raise ValueError(f"No usable feature columns found in {csv_path}")

    X = df[feature_cols].copy()
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models() -> dict[str, BaseEstimator]:
    return {
        "LogReg": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=N_JOBS,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=SEED,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=SEED,
        ),
        "XGBOOST": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=N_JOBS,
        ),
        "1DCNN": Torch1DCNNClassifier(
            epochs=8,
            batch_size=256,
            learning_rate=1e-3,
            dropout=0.3,
            patience=2,
            random_state=SEED,
        ),
    }


def build_scoring() -> dict[str, str | Any]:
    return {
        "Accuracy": "accuracy",
        "Balanced_Accuracy": "balanced_accuracy",
        "Precision": make_scorer(precision_score, zero_division=0),
        "Recall": make_scorer(recall_score, zero_division=0),
        "F1": make_scorer(f1_score, zero_division=0),
        "ROC_AUC": "roc_auc",
    }


def format_mean_std(mean_val: float, std_val: float) -> str:
    return f"{mean_val:.3f} +- {std_val:.3f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        cells = [str(value).replace("|", r"\|") for value in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def format_metric_columns(df: pd.DataFrame, columns: list[str], digits: int = 3) -> pd.DataFrame:
    formatted = df.copy()
    for col in columns:
        formatted[col] = formatted[col].map(lambda value: f"{float(value):.{digits}f}")
    return formatted


def safe_write_csv(df: pd.DataFrame, target_path: Path) -> Path:
    try:
        df.to_csv(target_path, index=False)
        return target_path
    except PermissionError:
        fallback_path = target_path.with_name(f"{target_path.stem}_latest{target_path.suffix}")
        df.to_csv(fallback_path, index=False)
        return fallback_path


def safe_write_text(content: str, target_path: Path) -> Path:
    try:
        target_path.write_text(content, encoding="utf-8")
        return target_path
    except PermissionError:
        fallback_path = target_path.with_name(f"{target_path.stem}_latest{target_path.suffix}")
        fallback_path.write_text(content, encoding="utf-8")
        return fallback_path


def run_cv_for_dataset(dataset_name: str, X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    preprocessor_template = build_preprocessor(X)
    models = build_models()
    scoring = build_scoring()
    metric_names = list(scoring.keys())

    fold_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | str]] = []

    for model_name, model in models.items():
        estimator = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor_template)),
                ("model", clone(model)),
            ]
        )

        cv_results = cross_validate(
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
            error_score="raise",
        )

        fold_metric_arrays: dict[str, np.ndarray] = {}
        for metric in metric_names:
            fold_metric_arrays[metric] = np.asarray(cv_results[f"test_{metric}"], dtype=float)

        for fold_idx in range(N_FOLDS):
            fold_rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Fold": fold_idx + 1,
                    "Accuracy": float(fold_metric_arrays["Accuracy"][fold_idx]),
                    "Balanced_Accuracy": float(fold_metric_arrays["Balanced_Accuracy"][fold_idx]),
                    "Precision": float(fold_metric_arrays["Precision"][fold_idx]),
                    "Recall": float(fold_metric_arrays["Recall"][fold_idx]),
                    "F1": float(fold_metric_arrays["F1"][fold_idx]),
                    "ROC_AUC": float(fold_metric_arrays["ROC_AUC"][fold_idx]),
                }
            )

        summary_rows.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy_mean": float(np.mean(fold_metric_arrays["Accuracy"])),
                "Accuracy_std": float(np.std(fold_metric_arrays["Accuracy"], ddof=1)),
                "Balanced_Accuracy_mean": float(np.mean(fold_metric_arrays["Balanced_Accuracy"])),
                "Balanced_Accuracy_std": float(np.std(fold_metric_arrays["Balanced_Accuracy"], ddof=1)),
                "Precision_mean": float(np.mean(fold_metric_arrays["Precision"])),
                "Precision_std": float(np.std(fold_metric_arrays["Precision"], ddof=1)),
                "Recall_mean": float(np.mean(fold_metric_arrays["Recall"])),
                "Recall_std": float(np.std(fold_metric_arrays["Recall"], ddof=1)),
                "F1_mean": float(np.mean(fold_metric_arrays["F1"])),
                "F1_std": float(np.std(fold_metric_arrays["F1"], ddof=1)),
                "ROC_AUC_mean": float(np.mean(fold_metric_arrays["ROC_AUC"])),
                "ROC_AUC_std": float(np.std(fold_metric_arrays["ROC_AUC"], ddof=1)),
            }
        )

    folds_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)

    summary_df["Accuracy_mean_std"] = summary_df.apply(
        lambda row: format_mean_std(row["Accuracy_mean"], row["Accuracy_std"]),
        axis=1,
    )
    summary_df["F1_mean_std"] = summary_df.apply(
        lambda row: format_mean_std(row["F1_mean"], row["F1_std"]),
        axis=1,
    )

    return folds_df, summary_df


def evaluate_models_on_holdout(
    dataset_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> pd.DataFrame:
    preprocessor_template = build_preprocessor(X_train)
    models = build_models()
    rows: list[dict[str, float | int | str]] = []

    for model_name, model in models.items():
        estimator = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor_template)),
                ("model", clone(model)),
            ]
        )
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        if hasattr(estimator, "predict_proba"):
            y_prob = estimator.predict_proba(X_test)[:, 1]
        elif hasattr(estimator, "decision_function"):
            decision = estimator.decision_function(X_test)
            y_prob = np.asarray(decision, dtype=float)
        else:
            y_prob = y_pred.astype(float)

        rows.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy": float(accuracy_score(y_test, y_pred)),
                "Balanced_Accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "F1": float(f1_score(y_test, y_pred, zero_division=0)),
                "ROC_AUC": float(roc_auc_score(y_test, y_prob)),
                "Train_Size": int(len(y_train)),
                "Final_Test_Size": int(len(y_test)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "Cognitive_Load-main" / "Cognitive_Load-main"
    output_root = Path(__file__).resolve().parent / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_files = {
        "real": data_root / "mental_health_wearable_data.csv",
        "synthetic": data_root / "synthetic_mental_health_data.csv",
    }

    all_folds: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    all_holdout: list[pd.DataFrame] = []
    split_rows: list[dict[str, int | str]] = []

    for dataset_name, csv_path in dataset_files.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset: {csv_path}")

        X, y = prepare_dataset(csv_path)

        X_train_val, X_final_test, y_train_val, y_final_test = train_test_split(
            X,
            y,
            test_size=FINAL_TEST_SIZE,
            stratify=y,
            random_state=SEED,
        )

        folds_df, summary_df = run_cv_for_dataset(dataset_name, X_train_val, y_train_val)
        holdout_df = evaluate_models_on_holdout(
            dataset_name=dataset_name,
            X_train=X_train_val,
            y_train=y_train_val,
            X_test=X_final_test,
            y_test=y_final_test,
        )
        all_folds.append(folds_df)
        all_summaries.append(summary_df)
        all_holdout.append(holdout_df)
        split_rows.append(
            {
                "Dataset": dataset_name,
                "Total_Size": int(len(y)),
                "Train_Val_Size": int(len(y_train_val)),
                "Final_Test_Size": int(len(y_final_test)),
            }
        )

    folds_all = pd.concat(all_folds, ignore_index=True)
    summary_all = pd.concat(all_summaries, ignore_index=True)
    holdout_all = pd.concat(all_holdout, ignore_index=True)
    split_summary = pd.DataFrame(split_rows)

    folds_csv = output_root / "cross_validation_fold_metrics.csv"
    summary_csv = output_root / "cross_validation_summary.csv"
    holdout_csv = output_root / "final_test_metrics.csv"
    split_csv = output_root / "data_split_summary.csv"
    rf_csv = output_root / "random_forest_accuracy_f1_mean_std.csv"
    fold_table_md = output_root / "cross_validation_fold_table.md"
    report_md = output_root / "cross_validation_report.md"

    saved_folds_csv = safe_write_csv(folds_all, folds_csv)
    saved_summary_csv = safe_write_csv(summary_all, summary_csv)
    saved_holdout_csv = safe_write_csv(holdout_all, holdout_csv)
    saved_split_csv = safe_write_csv(split_summary, split_csv)

    rf_summary = summary_all.loc[
        summary_all["Model"] == "RandomForest",
        ["Dataset", "Accuracy_mean_std", "F1_mean_std"],
    ].copy()
    saved_rf_csv = safe_write_csv(rf_summary, rf_csv)

    model_order = ["LogReg", "SVM", "RandomForest", "MLP", "XGBOOST", "1DCNN"]
    dataset_order = ["real", "synthetic"]
    metric_cols = ["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

    folds_display = folds_all.copy()
    folds_display["Dataset"] = pd.Categorical(folds_display["Dataset"], categories=dataset_order, ordered=True)
    folds_display["Model"] = pd.Categorical(folds_display["Model"], categories=model_order, ordered=True)
    folds_display = folds_display.sort_values(["Dataset", "Model", "Fold"]).reset_index(drop=True)
    folds_display = format_metric_columns(folds_display, metric_cols, digits=3)

    holdout_display = holdout_all.copy()
    holdout_display["Dataset"] = pd.Categorical(holdout_display["Dataset"], categories=dataset_order, ordered=True)
    holdout_display["Model"] = pd.Categorical(holdout_display["Model"], categories=model_order, ordered=True)
    holdout_display = holdout_display.sort_values(["Dataset", "Model"]).reset_index(drop=True)
    holdout_display = format_metric_columns(holdout_display, metric_cols, digits=3)

    fold_md_lines = [
        "# Fold-Level Cross-Validation Table",
        "",
        "Each row is one fold result for one model.",
        "",
    ]

    for dataset in dataset_order:
        subset = folds_display[folds_display["Dataset"] == dataset][
            ["Model", "Fold", "Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
        ]
        fold_md_lines.extend(
            [
                f"## {dataset.capitalize()} Dataset",
                "",
                dataframe_to_markdown(subset),
                "",
            ]
        )

    saved_fold_table_md = safe_write_text("\n".join(fold_md_lines), fold_table_md)

    lines = [
        "# Cross-Validation + Final Holdout Results",
        "",
        "Workflow follows scikit-learn guidance: split a final untouched test set, run CV on train/validation only, then evaluate once on the final test set.",
        "",
        "## Data Split Summary",
        "",
        dataframe_to_markdown(split_summary),
        "",
        "## 5-Fold Cross-Validation Summary (Train/Validation Partition)",
        "",
        "Metrics are reported as mean +- std across stratified folds.",
        "",
        "## Random Forest (for paper table)",
        "",
        dataframe_to_markdown(rf_summary),
        "",
        "## All Model Summary",
        "",
        dataframe_to_markdown(
            summary_all[
                [
                    "Dataset",
                    "Model",
                    "Accuracy_mean_std",
                    "F1_mean_std",
                    "Balanced_Accuracy_mean",
                    "Balanced_Accuracy_std",
                    "ROC_AUC_mean",
                    "ROC_AUC_std",
                ]
            ]
        ),
        "",
        "## Final Untouched Test-Set Results",
        "",
        dataframe_to_markdown(holdout_display[["Dataset", "Model", *metric_cols, "Train_Size", "Final_Test_Size"]]),
        "",
        "## Fold-Level Table",
        "",
        "The full fold-by-fold table is available in `cross_validation_fold_table.md`.",
        "",
    ]
    saved_report_md = safe_write_text("\n".join(lines), report_md)

    print("Saved outputs:")
    print(f"- {saved_folds_csv}")
    print(f"- {saved_summary_csv}")
    print(f"- {saved_holdout_csv}")
    print(f"- {saved_split_csv}")
    print(f"- {saved_rf_csv}")
    print(f"- {saved_fold_table_md}")
    print(f"- {saved_report_md}")
    print("\nRandom Forest (Accuracy and F1):")
    print(rf_summary.to_string(index=False))


if __name__ == "__main__":
    main()
