"""Train and compare multiple virality classifiers using Doc2Vec vectors and Reddit metadata.

Models trained:
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression

Outputs per model:
- Confusion matrix PNG
- Saved .joblib pipeline

Comparison outputs:
- classification_metrics.csv  (one row per model)
- combined_roc_curve.png
- model_comparison_bar.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "outputs" / "models"
TABLE_DIR = ROOT / "outputs" / "tables"
for directory in [FIG_DIR, MODEL_DIR, TABLE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def build_preprocessor(
    vector_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """Shared preprocessor used by all models."""
    numeric_features = vector_cols + numeric_cols
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_cols,
            ),
        ]
    )


def evaluate(
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Run predictions and return a metrics dict."""
    y_pred = pipeline.predict(X_test)

    # SVC with probability=True supports predict_proba; otherwise use decision_function.
    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_score = pipeline.decision_function(X_test)

    roc_auc = roc_auc_score(y_test, y_score) if y_test.nunique() > 1 else float("nan")

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "n_train": None,   # filled in main
        "n_test": len(y_test),
        "positive_rate_test": float(y_test.mean()),
        # keep y_score for ROC plot — removed before saving to CSV
        "_y_score": y_score,
        "_y_pred": y_pred,
    }


def save_confusion_matrix(name: str, y_test: pd.Series, y_pred: np.ndarray) -> None:
    slug = name.lower().replace(" ", "_")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"confusion_matrix_{slug}.png", dpi=200)
    plt.close()


def save_combined_roc(results: list[dict], y_test: pd.Series) -> None:
    """Plot all models on one ROC curve for easy comparison."""
    if y_test.nunique() < 2:
        return
    plt.figure(figsize=(8, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["_y_score"])
        auc = r["roc_auc"]
        plt.plot(fpr, tpr, label=f"{r['model']} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison — All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "combined_roc_curve.png", dpi=200)
    plt.close()


def save_comparison_bar(metrics_df: pd.DataFrame) -> None:
    """Side-by-side bar chart comparing accuracy, F1, and ROC-AUC across models."""
    plot_metrics = ["accuracy", "f1", "roc_auc"]
    x = np.arange(len(metrics_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(plot_metrics):
        ax.bar(x + i * width, metrics_df[metric], width, label=metric.upper().replace("_", "-"))

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_df["model"], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Accuracy, F1, ROC-AUC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison_bar.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare Reddit virality classifiers.")
    parser.add_argument("--input", default=str(PROCESSED_DIR / "doc2vec_vectors.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    vector_cols = [c for c in df.columns if c.startswith("doc2vec_")]
    numeric_cols = [
        c for c in ["num_comments", "upvote_ratio", "title_length", "token_count",
                    "has_body_text", "has_url", "hour_posted"]
        if c in df.columns
    ]
    categorical_cols = [c for c in ["subreddit", "listing", "day_of_week"] if c in df.columns]

    feature_cols = vector_cols + numeric_cols + categorical_cols
    X = df[feature_cols]
    y = df["viral"].astype(int)

    if y.nunique() < 2:
        raise RuntimeError("The target variable has only one class. Collect more posts or subreddits.")

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    # ------------------------------------------------------------------
    # Define all three models
    # ------------------------------------------------------------------
    model_definitions = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,   # enables predict_proba for ROC curve
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
            n_jobs=-1,
        ),
    }

    # ------------------------------------------------------------------
    # Train, evaluate, and save each model
    # ------------------------------------------------------------------
    results = []

    for name, clf in model_definitions.items():
        print(f"\nTraining {name}...")
        preprocessor = build_preprocessor(vector_cols, numeric_cols, categorical_cols)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)

        result = evaluate(name, pipeline, X_test, y_test)
        result["n_train"] = len(X_train)
        results.append(result)

        save_confusion_matrix(name, y_test, result["_y_pred"])

        slug = name.lower().replace(" ", "_")
        joblib.dump(pipeline, MODEL_DIR / f"virality_classifier_{slug}.joblib")
        print(f"  Accuracy={result['accuracy']:.3f}  F1={result['f1']:.3f}  ROC-AUC={result['roc_auc']:.3f}")

    # ------------------------------------------------------------------
    # Comparison outputs
    # ------------------------------------------------------------------
    save_combined_roc(results, y_test)
    save_comparison_bar(pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")} for r in results
    ]))

    metrics_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")} for r in results
    ])
    metrics_df.to_csv(TABLE_DIR / "classification_metrics.csv", index=False)

    print("\n=== Model Comparison ===")
    print(metrics_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
    print(f"\nFigures saved to {FIG_DIR}")
    print(f"Models saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
