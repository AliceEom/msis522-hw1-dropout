#!/usr/bin/env python3
"""
MSIS 522 HW1 end-to-end training pipeline.

Implements:
- Train-only feature recheck (18-feature baseline + top-10 comparison)
- Descriptive analytics plots
- Logistic baseline, Decision Tree (GridSearchCV), Random Forest (GridSearchCV),
  LightGBM (GridSearchCV), and Keras MLP
- Bonus MLP hyperparameter tuning with visualization
- SHAP explainability (summary, bar, waterfall) for best tree-based model
- Saved artifacts for Streamlit deployment
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_msis522_hw1")

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
import tensorflow as tf
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight


RANDOM_STATE = 42

BASE_FEATURES_18 = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Mother's qualification",
    "Admission grade",
    "Displaced",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "GDP",
]

MANUAL_INPUT_FEATURES = [
    "Age at enrollment",
    "Admission grade",
    "Curricular units 1st sem (grade)",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "Application order",
    "GDP",
]


@dataclass
class Paths:
    root: Path
    data_csv: Path
    artifacts: Path
    models: Path
    figures: Path
    metrics: Path
    metadata: Path
    logs: Path


def build_paths(root: Path) -> Paths:
    artifacts = root / "artifacts"
    return Paths(
        root=root,
        data_csv=root / "data" / "studentdata_raw.csv",
        artifacts=artifacts,
        models=artifacts / "models",
        figures=artifacts / "figures",
        metrics=artifacts / "metrics",
        metadata=artifacts / "metadata",
        logs=artifacts / "logs",
    )


def ensure_dirs(paths: Paths) -> None:
    for p in [
        paths.artifacts,
        paths.models,
        paths.figures,
        paths.metrics,
        paths.metadata,
        paths.logs,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    mpl_dir = paths.logs / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    matplotlib.use("Agg")


def set_seed(seed: int = RANDOM_STATE) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace('"', "", regex=False)
        .str.replace("\t", "", regex=False)
    )
    return df


def load_data(data_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(data_csv, sep=";", quoting=3)
    df = clean_columns(df)

    df["Target"] = df["Target"].astype(str).str.strip()
    df["Target_label"] = df["Target"]
    df["Dropout_flag"] = (df["Target"] == "Dropout").astype(int)

    predictor_cols = [c for c in df.columns if c not in {"Target", "Target_label", "Dropout_flag"}]
    for c in predictor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def univariate_logit_screen(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    y = train_df["Dropout_flag"].astype(int)
    pvals: Dict[str, float] = {}

    for col in feature_cols:
        x = train_df[col].astype(float)
        if x.isna().all() or x.nunique(dropna=True) < 2:
            pvals[col] = 1.0
            continue

        x = x.fillna(x.median())
        X_uni = sm.add_constant(x, has_constant="add")
        try:
            model = sm.Logit(y, X_uni).fit(disp=0)
            pvals[col] = float(model.pvalues.get(col, 1.0))
        except Exception:
            pvals[col] = 1.0

    return pvals


def corr_filter_by_pvalue(
    train_df: pd.DataFrame,
    candidate_features: List[str],
    pvals: Dict[str, float],
    threshold: float = 0.75,
) -> List[str]:
    usable = [f for f in candidate_features if f in train_df.columns]
    if not usable:
        return []

    corr = train_df[usable].corr().abs().fillna(0.0)
    ordered = sorted(usable, key=lambda f: pvals.get(f, 1.0))
    selected: List[str] = []

    for f in ordered:
        if not selected:
            selected.append(f)
            continue
        max_corr = max(float(corr.loc[f, s]) for s in selected)
        if max_corr <= threshold:
            selected.append(f)

    return selected


def build_feature_sets(
    train_df: pd.DataFrame,
    all_predictors: List[str],
    base_18: List[str],
) -> Dict[str, Any]:
    pvals = univariate_logit_screen(train_df, all_predictors)

    significant_pool = [f for f in all_predictors if pvals.get(f, 1.0) < 0.05]
    significant_pool = sorted(significant_pool, key=lambda f: pvals[f])

    base_significant = [f for f in base_18 if pvals.get(f, 1.0) < 0.05]
    selected_18 = corr_filter_by_pvalue(train_df, base_significant, pvals, threshold=0.75)

    # Fill back to 18 with train-screened significant features in p-value priority order.
    for f in significant_pool:
        if len(selected_18) >= 18:
            break
        if f not in selected_18:
            selected_18.append(f)

    # Final fallback: keep base list order to guarantee 18 features.
    for f in base_18:
        if len(selected_18) >= 18:
            break
        if f not in selected_18:
            selected_18.append(f)

    selected_18 = selected_18[:18]

    return {
        "pvalues": pvals,
        "significant_pool": significant_pool,
        "selected_18": selected_18,
    }


def get_logistic_pipeline(feature_names: List[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_names,
            )
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def get_tree_preprocess(feature_names: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                feature_names,
            )
        ]
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def evaluate_sklearn_classifier(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.values, y_pred, y_prob)
    return metrics, y_pred, y_prob


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def create_part1_figures(df: pd.DataFrame, feature_names: List[str], paths: Paths) -> Dict[str, str]:
    fig_map: Dict[str, str] = {}

    # 1) Target distribution
    counts = df["Dropout_flag"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Non-dropout (0)", "Dropout (1)"], y=counts.values, palette=["#4C72B0", "#C44E52"])
    plt.title("Target Distribution (Binary Dropout)")
    plt.ylabel("Count")
    plt.tight_layout()
    out = paths.figures / "part1_target_distribution.png"
    plt.savefig(out, dpi=160)
    plt.close()
    fig_map["target_distribution"] = str(out)

    # 2) Admission grade by outcome
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x="Dropout_flag", y="Admission grade", palette=["#55A868", "#C44E52"])
    plt.xticks([0, 1], ["Non-dropout", "Dropout"])
    plt.title("Admission Grade by Outcome")
    plt.tight_layout()
    out = paths.figures / "part1_admission_grade_boxplot.png"
    plt.savefig(out, dpi=160)
    plt.close()
    fig_map["admission_grade_boxplot"] = str(out)

    # 3) First-sem grade by outcome
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(
        data=df,
        x="Dropout_flag",
        y="Curricular units 1st sem (grade)",
        palette=["#55A868", "#C44E52"],
    )
    plt.xticks([0, 1], ["Non-dropout", "Dropout"])
    plt.title("1st Semester Grade by Outcome")
    plt.tight_layout()
    out = paths.figures / "part1_first_sem_grade_boxplot.png"
    plt.savefig(out, dpi=160)
    plt.close()
    fig_map["first_sem_grade_boxplot"] = str(out)

    # 4) Dropout rate by grade quartile
    tmp = df[["Curricular units 1st sem (grade)", "Dropout_flag"]].dropna().copy()
    tmp["grade_q"] = pd.qcut(
        tmp["Curricular units 1st sem (grade)"],
        4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        duplicates="drop",
    )
    dropout_by_q = tmp.groupby("grade_q", observed=True)["Dropout_flag"].mean().reset_index()
    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=dropout_by_q, x="grade_q", y="Dropout_flag", palette="viridis")
    plt.title("Dropout Rate by 1st Semester Grade Quartile")
    plt.ylabel("Dropout Rate")
    plt.ylim(0, min(1.0, dropout_by_q["Dropout_flag"].max() + 0.1))
    plt.tight_layout()
    out = paths.figures / "part1_dropout_by_grade_quartile.png"
    plt.savefig(out, dpi=160)
    plt.close()
    fig_map["dropout_by_grade_quartile"] = str(out)

    # 5) Correlation heatmap (features + target)
    corr_cols = list(dict.fromkeys(feature_names + ["Dropout_flag"]))
    corr = df[corr_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap (Selected Features + Target)")
    plt.tight_layout()
    out = paths.figures / "part1_correlation_heatmap.png"
    plt.savefig(out, dpi=160)
    plt.close()
    fig_map["correlation_heatmap"] = str(out)

    return fig_map


def compare_18_vs_10(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_18: List[str],
) -> Dict[str, Any]:
    logistic_18 = get_logistic_pipeline(selected_18)
    logistic_18.fit(X_train[selected_18], y_train)

    coefs = np.abs(logistic_18.named_steps["model"].coef_[0])
    coef_df = pd.DataFrame({"feature": selected_18, "abs_coef": coefs}).sort_values("abs_coef", ascending=False)
    selected_10 = coef_df["feature"].head(10).tolist()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_f1_18 = cross_val_score(logistic_18, X_train[selected_18], y_train, cv=cv, scoring="f1", n_jobs=1)

    logistic_10 = get_logistic_pipeline(selected_10)
    cv_f1_10 = cross_val_score(logistic_10, X_train[selected_10], y_train, cv=cv, scoring="f1", n_jobs=1)

    mean18 = float(cv_f1_18.mean())
    mean10 = float(cv_f1_10.mean())
    final_features = selected_10 if mean10 > mean18 else selected_18

    return {
        "selected_10": selected_10,
        "cv_f1_18": mean18,
        "cv_f1_10": mean10,
        "final_features": final_features,
    }


def build_keras_mlp(input_dim: int, hidden_layers: Tuple[int, int], dropout_rate: float, learning_rate: float) -> tf.keras.Model:
    model = tf.keras.Sequential(name="mlp_dropout")
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(hidden_layers[0], activation="relu"))
    if dropout_rate > 0:
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(hidden_layers[1], activation="relu"))
    if dropout_rate > 0:
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    optimizer_cls = (
        tf.keras.optimizers.legacy.Adam
        if hasattr(tf.keras.optimizers, "legacy")
        else tf.keras.optimizers.Adam
    )

    model.compile(
        optimizer=optimizer_cls(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def extract_pos_class_shap(shap_values: Any, expected_value: Any) -> Tuple[np.ndarray, float]:
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        if isinstance(expected_value, (list, np.ndarray)):
            base = float(np.array(expected_value).ravel()[min(1, len(np.array(expected_value).ravel()) - 1)])
        else:
            base = float(expected_value)
        return np.array(values), base

    values = np.array(shap_values)
    if values.ndim == 3:
        values = values[:, :, 1]

    if isinstance(expected_value, (list, np.ndarray)):
        base_arr = np.array(expected_value).ravel()
        base = float(base_arr[min(1, len(base_arr) - 1)])
    else:
        base = float(expected_value)
    return values, base


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    root = Path(__file__).resolve().parent
    paths = build_paths(root)
    ensure_dirs(paths)
    set_seed(RANDOM_STATE)

    print("[1/7] Loading data...")
    df = load_data(paths.data_csv)

    dataset_stats = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "target_counts_binary": {str(k): int(v) for k, v in df["Dropout_flag"].value_counts().to_dict().items()},
        "target_ratio_binary": {str(k): float(v) for k, v in df["Dropout_flag"].value_counts(normalize=True).round(6).to_dict().items()},
        "target_counts_original": {str(k): int(v) for k, v in df["Target"].value_counts().to_dict().items()},
    }

    all_predictors = [c for c in df.columns if c not in {"Target", "Target_label", "Dropout_flag"}]

    print("[2/7] Train/test split + train-only feature recheck...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["Dropout_flag"],
        random_state=RANDOM_STATE,
    )

    feature_info = build_feature_sets(train_df, all_predictors, BASE_FEATURES_18)
    selected_18 = feature_info["selected_18"]

    compare_info = compare_18_vs_10(
        X_train=train_df,
        y_train=train_df["Dropout_flag"],
        selected_18=selected_18,
    )

    final_features = compare_info["final_features"]

    # Build part-1 figures using final feature set.
    print("[3/7] Creating Part 1 figures...")
    part1_figs = create_part1_figures(df, final_features, paths)

    # Prepare split matrices.
    X_train = train_df[final_features].copy()
    y_train = train_df["Dropout_flag"].astype(int).copy()
    X_test = test_df[final_features].copy()
    y_test = test_df["Dropout_flag"].astype(int).copy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    model_metrics: Dict[str, Dict[str, Any]] = {}
    y_prob_store: Dict[str, np.ndarray] = {}
    best_params: Dict[str, Dict[str, Any]] = {}
    model_store: Dict[str, Any] = {}

    print("[4/7] Training/tuning Part 2 models...")
    # 2.2 Logistic baseline
    logistic_pipe = get_logistic_pipeline(final_features)
    logistic_pipe.fit(X_train, y_train)
    metrics_logit, _, prob_logit = evaluate_sklearn_classifier(logistic_pipe, X_test, y_test)
    model_metrics["logistic"] = metrics_logit
    y_prob_store["logistic"] = prob_logit
    best_params["logistic"] = {
        "model": "LogisticRegression",
        "class_weight": "balanced",
        "solver": "liblinear",
        "max_iter": 5000,
    }
    model_store["logistic"] = logistic_pipe
    joblib.dump(logistic_pipe, paths.models / "logistic_pipeline.joblib")

    # 2.3 Decision Tree / CART with GridSearchCV
    tree_pipe = Pipeline(
        steps=[
            ("preprocess", get_tree_preprocess(final_features)),
            (
                "model",
                DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            ),
        ]
    )
    tree_grid = {
        "model__max_depth": [3, 5, 7, 10],
        "model__min_samples_leaf": [5, 10, 20, 50],
    }
    gs_tree = GridSearchCV(tree_pipe, tree_grid, scoring="f1", cv=cv, n_jobs=1, refit=True)
    gs_tree.fit(X_train, y_train)
    best_tree = gs_tree.best_estimator_
    metrics_tree, _, prob_tree = evaluate_sklearn_classifier(best_tree, X_test, y_test)
    model_metrics["decision_tree"] = metrics_tree
    y_prob_store["decision_tree"] = prob_tree
    best_params["decision_tree"] = gs_tree.best_params_
    model_store["decision_tree"] = best_tree
    joblib.dump(best_tree, paths.models / "decision_tree_pipeline.joblib")

    # Tree visualization
    X_train_tree = best_tree.named_steps["preprocess"].transform(X_train)
    plt.figure(figsize=(16, 8))
    plot_tree(
        best_tree.named_steps["model"],
        feature_names=final_features,
        class_names=["Non-dropout", "Dropout"],
        filled=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title("Best Decision Tree (Top Levels)")
    plt.tight_layout()
    plt.savefig(paths.figures / "part2_best_decision_tree.png", dpi=170)
    plt.close()

    # 2.4 Random Forest with GridSearchCV
    rf_pipe = Pipeline(
        steps=[
            ("preprocess", get_tree_preprocess(final_features)),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    rf_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 5, 8],
    }
    gs_rf = GridSearchCV(rf_pipe, rf_grid, scoring="f1", cv=cv, n_jobs=1, refit=True)
    gs_rf.fit(X_train, y_train)
    best_rf = gs_rf.best_estimator_
    metrics_rf, _, prob_rf = evaluate_sklearn_classifier(best_rf, X_test, y_test)
    model_metrics["random_forest"] = metrics_rf
    y_prob_store["random_forest"] = prob_rf
    best_params["random_forest"] = gs_rf.best_params_
    model_store["random_forest"] = best_rf
    joblib.dump(best_rf, paths.models / "random_forest_pipeline.joblib")

    # 2.5 Boosted Trees (LightGBM) with 3+ params
    lgbm_pipe = Pipeline(
        steps=[
            ("preprocess", get_tree_preprocess(final_features)),
            (
                "model",
                LGBMClassifier(
                    objective="binary",
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=1,
                    verbose=-1,
                ),
            ),
        ]
    )
    lgbm_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.05, 0.1],
    }
    gs_lgbm = GridSearchCV(lgbm_pipe, lgbm_grid, scoring="f1", cv=cv, n_jobs=1, refit=True)
    gs_lgbm.fit(X_train, y_train)
    best_lgbm = gs_lgbm.best_estimator_
    metrics_lgbm, _, prob_lgbm = evaluate_sklearn_classifier(best_lgbm, X_test, y_test)
    model_metrics["lightgbm"] = metrics_lgbm
    y_prob_store["lightgbm"] = prob_lgbm
    best_params["lightgbm"] = gs_lgbm.best_params_
    model_store["lightgbm"] = best_lgbm
    joblib.dump(best_lgbm, paths.models / "lightgbm_pipeline.joblib")

    # 2.6 MLP with Keras + Bonus tuning grid
    mlp_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_train_scaled = mlp_preprocess.fit_transform(X_train)
    X_test_scaled = mlp_preprocess.transform(X_test)

    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train_scaled,
        y_train.values,
        test_size=0.20,
        stratify=y_train.values,
        random_state=RANDOM_STATE,
    )

    classes = np.unique(y_sub)
    class_weights_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y_sub)
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_vals)}

    tuning_grid = {
        "hidden_layers": [(64, 64), (128, 128), (128, 64)],
        "learning_rate": [0.001, 0.0005],
        "dropout_rate": [0.0, 0.2],
    }

    tuning_results = []
    best_combo = None
    best_val_f1 = -1.0

    for hidden in tuning_grid["hidden_layers"]:
        for lr in tuning_grid["learning_rate"]:
            for dr in tuning_grid["dropout_rate"]:
                tf.keras.backend.clear_session()
                model = build_keras_mlp(
                    input_dim=X_train_scaled.shape[1],
                    hidden_layers=hidden,
                    dropout_rate=dr,
                    learning_rate=lr,
                )

                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=6,
                        restore_best_weights=True,
                    )
                ]

                history = model.fit(
                    X_sub,
                    y_sub,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=64,
                    verbose=0,
                    callbacks=callbacks,
                    class_weight=class_weights,
                )

                val_prob = model.predict(X_val, verbose=0).ravel()
                val_pred = (val_prob >= 0.5).astype(int)
                val_f1 = float(f1_score(y_val, val_pred, zero_division=0))

                result = {
                    "hidden_layers": str(hidden),
                    "learning_rate": float(lr),
                    "dropout_rate": float(dr),
                    "val_f1": val_f1,
                    "epochs_trained": int(len(history.history["loss"])),
                    "best_val_loss": float(min(history.history["val_loss"])),
                }
                tuning_results.append(result)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_combo = (hidden, lr, dr)

    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(paths.metrics / "bonus_mlp_tuning_results.csv", index=False)

    # Bonus visualization heatmap
    tuning_df["combo"] = tuning_df.apply(
        lambda r: f"lr={r['learning_rate']:.4f}\ndr={r['dropout_rate']:.1f}", axis=1
    )
    heat = tuning_df.pivot_table(index="hidden_layers", columns="combo", values="val_f1", aggfunc="max")
    plt.figure(figsize=(8, 4.8))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Bonus: MLP Hyperparameter Tuning (Validation F1)")
    plt.tight_layout()
    plt.savefig(paths.figures / "bonus_mlp_tuning_heatmap.png", dpi=170)
    plt.close()

    # Refit best MLP on full training set for final metrics.
    assert best_combo is not None
    best_hidden, best_lr, best_dr = best_combo

    tf.keras.backend.clear_session()
    mlp_model = build_keras_mlp(
        input_dim=X_train_scaled.shape[1],
        hidden_layers=best_hidden,
        dropout_rate=best_dr,
        learning_rate=best_lr,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        )
    ]

    history = mlp_model.fit(
        X_train_scaled,
        y_train.values,
        validation_split=0.20,
        epochs=60,
        batch_size=64,
        verbose=0,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    mlp_prob = mlp_model.predict(X_test_scaled, verbose=0).ravel()
    mlp_pred = (mlp_prob >= 0.5).astype(int)
    metrics_mlp = compute_metrics(y_test.values, mlp_pred, mlp_prob)
    model_metrics["mlp_keras"] = metrics_mlp
    y_prob_store["mlp_keras"] = mlp_prob
    best_params["mlp_keras"] = {
        "hidden_layers": str(best_hidden),
        "learning_rate": float(best_lr),
        "dropout_rate": float(best_dr),
        "best_validation_f1": float(best_val_f1),
    }

    mlp_model.save(paths.models / "mlp_keras_model.keras")
    joblib.dump({"preprocess": mlp_preprocess, "features": final_features}, paths.models / "mlp_preprocess.joblib")

    # MLP history plot
    plt.figure(figsize=(8, 4.8))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    if "accuracy" in history.history:
        plt.plot(history.history["accuracy"], label="train_acc", linestyle="--")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val_acc", linestyle="--")
    plt.title("MLP Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.figures / "part2_mlp_training_history.png", dpi=170)
    plt.close()

    # Save ROC plots for all models.
    for model_name, probs in y_prob_store.items():
        save_roc_curve(
            y_true=y_test.values,
            y_prob=probs,
            title=f"ROC Curve — {model_name}",
            out_path=paths.figures / f"part2_roc_{model_name}.png",
        )

    # 2.7 comparison table + bar chart
    comparison_rows = []
    for model_name, metric_dict in model_metrics.items():
        row = {"model": model_name}
        row.update(metric_dict)
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows).sort_values("f1", ascending=False)
    comparison_df.to_csv(paths.metrics / "part2_model_comparison.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=comparison_df, x="model", y="f1", palette="Set2")
    plt.title("Model Comparison by F1 Score")
    plt.xlabel("Model")
    plt.ylabel("F1")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths.figures / "part2_f1_bar_comparison.png", dpi=170)
    plt.close()

    # Choose best tree-based model for SHAP.
    tree_candidates = ["decision_tree", "random_forest", "lightgbm"]
    tree_best_name = max(tree_candidates, key=lambda n: model_metrics[n]["f1"])
    tree_best_pipe = model_store[tree_best_name]

    print(f"[5/7] SHAP analysis on best tree-based model: {tree_best_name}")
    X_test_tree = pd.DataFrame(
        tree_best_pipe.named_steps["preprocess"].transform(X_test),
        columns=final_features,
    )
    tree_model = tree_best_pipe.named_steps["model"]

    explainer = shap.TreeExplainer(tree_model)
    shap_raw = explainer.shap_values(X_test_tree)
    shap_values, shap_base = extract_pos_class_shap(shap_raw, explainer.expected_value)

    # SHAP summary beeswarm
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test_tree, feature_names=final_features, show=False)
    plt.tight_layout()
    plt.savefig(paths.figures / "part3_shap_summary_beeswarm.png", dpi=170, bbox_inches="tight")
    plt.close()

    # SHAP bar plot
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test_tree, feature_names=final_features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(paths.figures / "part3_shap_bar.png", dpi=170, bbox_inches="tight")
    plt.close()

    # SHAP waterfall for one high-risk case.
    high_risk_idx = int(np.argmax(y_prob_store[tree_best_name]))
    explanation = shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=shap_base,
        data=X_test_tree.iloc[high_risk_idx].values,
        feature_names=final_features,
    )
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(explanation, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(paths.figures / "part3_shap_waterfall_example.png", dpi=170, bbox_inches="tight")
    plt.close()

    # Save full metadata + artifacts index.
    print("[6/7] Saving metadata and report artifacts...")
    feature_ranges = {}
    feature_means = {}
    for f in final_features:
        series = df[f].dropna()
        feature_ranges[f] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
        }
        feature_means[f] = float(series.mean())

    # Save params / metrics json.
    save_json(paths.metrics / "part2_metrics.json", model_metrics)
    save_json(paths.metrics / "part2_best_params.json", best_params)

    captions = {
        "target_distribution": (
            "Dropout and non-dropout classes are not balanced, so accuracy alone is not enough for evaluation. "
            "We prioritize F1 and AUC to avoid over-crediting majority-class predictions."
        ),
        "admission_grade_boxplot": (
            "Admission grade is generally lower in the dropout group, consistent with the earlier project narrative. "
            "This supports using academic readiness as an early risk signal."
        ),
        "first_sem_grade_boxplot": (
            "1st-semester grade separates outcomes clearly and appears as the strongest practical predictor. "
            "This aligns with the previous notebook insight that early academic performance drives risk stratification."
        ),
        "dropout_by_grade_quartile": (
            "Students in the lowest first-semester quartile have substantially higher dropout risk than the upper quartiles. "
            "This nonlinear risk jump justifies targeted early intervention for Q1 students."
        ),
        "correlation_heatmap": (
            "Several academic variables are strongly correlated, which is why correlation filtering was reapplied on train only. "
            "Reducing collinearity improves model stability and interpretability."
        ),
    }

    artifact_index = {
        "part1_figures": part1_figs,
        "part2_figures": {
            "decision_tree": str(paths.figures / "part2_best_decision_tree.png"),
            "f1_bar": str(paths.figures / "part2_f1_bar_comparison.png"),
            "mlp_history": str(paths.figures / "part2_mlp_training_history.png"),
            "bonus_mlp_heatmap": str(paths.figures / "bonus_mlp_tuning_heatmap.png"),
            "roc_logistic": str(paths.figures / "part2_roc_logistic.png"),
            "roc_decision_tree": str(paths.figures / "part2_roc_decision_tree.png"),
            "roc_random_forest": str(paths.figures / "part2_roc_random_forest.png"),
            "roc_lightgbm": str(paths.figures / "part2_roc_lightgbm.png"),
            "roc_mlp_keras": str(paths.figures / "part2_roc_mlp_keras.png"),
        },
        "part3_figures": {
            "shap_summary": str(paths.figures / "part3_shap_summary_beeswarm.png"),
            "shap_bar": str(paths.figures / "part3_shap_bar.png"),
            "shap_waterfall": str(paths.figures / "part3_shap_waterfall_example.png"),
        },
    }

    metadata = {
        "random_state": RANDOM_STATE,
        "dataset_stats": dataset_stats,
        "feature_selection": {
            "base_18": BASE_FEATURES_18,
            "selected_18_rechecked": selected_18,
            "selected_10": compare_info["selected_10"],
            "cv_f1_18": compare_info["cv_f1_18"],
            "cv_f1_10": compare_info["cv_f1_10"],
            "final_features": final_features,
            "pvalues": feature_info["pvalues"],
        },
        "final_model_choice_logic": "Use higher train 5-fold CV F1 between 18-feature and 10-feature sets; choose 18 on tie.",
        "manual_input_features": [f for f in MANUAL_INPUT_FEATURES if f in final_features],
        "feature_ranges": feature_ranges,
        "feature_means": feature_means,
        "best_tree_model_for_shap": tree_best_name,
        "captions": captions,
        "artifact_index": artifact_index,
        "best_params": best_params,
    }

    save_json(paths.metadata / "project_metadata.json", metadata)

    # Save model comparison paragraph draft (reused style from prior project).
    comparison_text = (
        "Across models, tree ensembles capture nonlinear dropout patterns better than linear baselines, "
        "while logistic regression remains useful for interpretability through signed effects. "
        "The final recommendation balances F1/AUC performance with stakeholder explainability, and SHAP is used to show local and global impact directions."
    )
    (paths.metadata / "model_comparison_paragraph.txt").write_text(comparison_text, encoding="utf-8")

    print("[7/7] Done.")
    print(f"Artifacts saved under: {paths.artifacts}")


if __name__ == "__main__":
    main()
