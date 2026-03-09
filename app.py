#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_msis522_hw1_app")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
import tensorflow as tf
import altair as alt
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Dropout Risk Modeling", layout="wide")

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
METRICS = ARTIFACTS / "metrics"
MODELS = ARTIFACTS / "models"
META = ARTIFACTS / "metadata"
DATA = ROOT / "data" / "studentdata_raw.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, sep=";", quoting=3)
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace('"', "", regex=False)
        .str.replace("\t", "", regex=False)
    )
    df["Target"] = df["Target"].astype(str).str.strip()
    df["Dropout_flag"] = (df["Target"] == "Dropout").astype(int)
    return df


def strongest_corr_pair(df: pd.DataFrame, feature_names: List[str]) -> Tuple[str, str, float]:
    corr = df[feature_names].corr().abs()
    arr = corr.to_numpy()
    np.fill_diagonal(arr, np.nan)
    if np.isnan(arr).all():
        return "N/A", "N/A", float("nan")
    i, j = np.unravel_index(np.nanargmax(arr), arr.shape)
    return str(corr.index[i]), str(corr.columns[j]), float(arr[i, j])


def get_corr_value(corr_df: pd.DataFrame, col_a: str, col_b: str) -> float:
    if col_a in corr_df.columns and col_b in corr_df.columns:
        return float(corr_df.loc[col_a, col_b])
    return float("nan")


def get_full_corr_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != "Dropout_flag"]
    if "Dropout_flag" in df.columns:
        cols.append("Dropout_flag")
    return cols


def get_top_target_corr_columns(corr_df: pd.DataFrame, target_col: str = "Dropout_flag", top_k: int = 12) -> List[str]:
    if target_col not in corr_df.columns:
        return [target_col]
    target_abs = corr_df[target_col].drop(labels=[target_col], errors="ignore").abs().sort_values(ascending=False)
    top_cols = target_abs.head(top_k).index.tolist()
    return top_cols + [target_col]


def make_dropout_rate_figure(
    df: pd.DataFrame,
    feature: str,
    labels: Dict[int, str],
    title: str,
) -> plt.Figure:
    tmp = df[[feature, "Dropout_flag"]].dropna().copy()
    rate = tmp.groupby(feature, observed=True)["Dropout_flag"].mean().reset_index()
    rate["label"] = rate[feature].astype(int).map(labels).fillna(rate[feature].astype(str))

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    sns.barplot(data=rate, x="label", y="Dropout_flag", palette="Set2", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Dropout Rate")
    ax.set_ylim(0, min(1.0, float(rate["Dropout_flag"].max()) + 0.15))
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def make_grade_band_dropout_figure(df: pd.DataFrame) -> plt.Figure:
    tmp = df[["Curricular units 1st sem (grade)", "Dropout_flag"]].dropna().copy()
    tmp["grade_band"] = pd.qcut(
        tmp["Curricular units 1st sem (grade)"],
        4,
        labels=[
            "Lowest 25% grades",
            "Lower-middle 25%",
            "Upper-middle 25%",
            "Highest 25% grades",
        ],
        duplicates="drop",
    )
    rate = tmp.groupby("grade_band", observed=True)["Dropout_flag"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    sns.barplot(data=rate, x="grade_band", y="Dropout_flag", palette="viridis", ax=ax)
    ax.set_title("Dropout Rate by 1st-Semester Grade Bands")
    ax.set_xlabel("")
    ax.set_ylabel("Dropout Rate")
    ax.set_ylim(0, min(1.0, float(rate["Dropout_flag"].max()) + 0.1))
    ax.grid(axis="y", alpha=0.25)
    for label in ax.get_xticklabels():
        label.set_rotation(12)
        label.set_ha("right")
    plt.tight_layout()
    return fig


def make_second_sem_violin_figure(df: pd.DataFrame) -> plt.Figure:
    tmp = df[["Dropout_flag", "Curricular units 2nd sem (grade)"]].dropna().copy()
    tmp["Outcome"] = tmp["Dropout_flag"].map({0: "Non-dropout", 1: "Dropout"})
    order = ["Non-dropout", "Dropout"]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    sns.violinplot(
        data=tmp,
        x="Outcome",
        y="Curricular units 2nd sem (grade)",
        order=order,
        density_norm="area",
        inner="box",
        bw_adjust=1.0,
        cut=0,
        palette={"Non-dropout": "#4C78A8", "Dropout": "#F58518"},
        ax=ax,
    )
    means = tmp.groupby("Outcome", observed=True)["Curricular units 2nd sem (grade)"].mean()
    for xpos, grp in enumerate(order):
        if grp in means.index and not np.isnan(float(means.loc[grp])):
            y = float(means.loc[grp])
            ax.scatter(xpos, y, s=45, c="black", zorder=5, marker="o")
            ax.text(xpos + 0.04, y + 0.12, f"mean={y:.2f}", fontsize=9, color="black")
    ax.set_title("2nd-Semester Grade Distribution by Outcome (Violin Plot)")
    ax.set_xlabel("")
    ax.set_ylabel("2nd-Semester Grade")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def make_full_correlation_heatmap_figure(df: pd.DataFrame, corr_cols: List[str]) -> plt.Figure:
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(24, 20))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 4},
        cbar_kws={"shrink": 0.8},
        linewidths=0.15,
        linecolor="white",
        xticklabels=True,
        yticklabels=True,
        ax=ax,
    )
    ax.set_title("Full Correlation Heatmap (All Numeric Columns)")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    plt.tight_layout()
    return fig


def make_focused_correlation_heatmap_figure(df: pd.DataFrame, corr_cols: List[str]) -> plt.Figure:
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.85},
        linewidths=0.2,
        linecolor="white",
        xticklabels=True,
        yticklabels=True,
        ax=ax,
    )
    ax.set_title("Focused Correlation Heatmap (Top Target-linked Variables)")
    wrapped_x = [
        textwrap.fill(str(lbl), width=16, break_long_words=False, break_on_hyphens=False)
        for lbl in corr.columns
    ]
    wrapped_y = [
        textwrap.fill(str(lbl), width=24, break_long_words=False, break_on_hyphens=False)
        for lbl in corr.index
    ]
    ax.set_xticklabels(wrapped_x, rotation=0, fontsize=8, ha="center")
    ax.set_yticklabels(wrapped_y, rotation=0, fontsize=8)
    plt.subplots_adjust(bottom=0.30, left=0.35)
    plt.tight_layout()
    return fig


@st.cache_data
def compute_eda_highlights(df: pd.DataFrame) -> Dict[str, float]:
    tmp = df.copy()
    out: Dict[str, float] = {}

    out["dropout_rate"] = float(tmp["Dropout_flag"].mean())

    debtor_1 = tmp.loc[tmp["Debtor"] == 1, "Dropout_flag"]
    debtor_0 = tmp.loc[tmp["Debtor"] == 0, "Dropout_flag"]
    out["dropout_debtor_1"] = float(debtor_1.mean()) if len(debtor_1) else float("nan")
    out["dropout_debtor_0"] = float(debtor_0.mean()) if len(debtor_0) else float("nan")

    sch_1 = tmp.loc[tmp["Scholarship holder"] == 1, "Dropout_flag"]
    sch_0 = tmp.loc[tmp["Scholarship holder"] == 0, "Dropout_flag"]
    out["dropout_scholar_1"] = float(sch_1.mean()) if len(sch_1) else float("nan")
    out["dropout_scholar_0"] = float(sch_0.mean()) if len(sch_0) else float("nan")

    tuition_1 = tmp.loc[tmp["Tuition fees up to date"] == 1, "Dropout_flag"]
    tuition_0 = tmp.loc[tmp["Tuition fees up to date"] == 0, "Dropout_flag"]
    out["dropout_tuition_1"] = float(tuition_1.mean()) if len(tuition_1) else float("nan")
    out["dropout_tuition_0"] = float(tuition_0.mean()) if len(tuition_0) else float("nan")

    out["admission_non_dropout"] = float(tmp.loc[tmp["Dropout_flag"] == 0, "Admission grade"].mean())
    out["admission_dropout"] = float(tmp.loc[tmp["Dropout_flag"] == 1, "Admission grade"].mean())

    out["first_sem_non_dropout"] = float(tmp.loc[tmp["Dropout_flag"] == 0, "Curricular units 1st sem (grade)"].mean())
    out["first_sem_dropout"] = float(tmp.loc[tmp["Dropout_flag"] == 1, "Curricular units 1st sem (grade)"].mean())
    out["second_sem_non_dropout"] = float(tmp.loc[tmp["Dropout_flag"] == 0, "Curricular units 2nd sem (grade)"].mean())
    out["second_sem_dropout"] = float(tmp.loc[tmp["Dropout_flag"] == 1, "Curricular units 2nd sem (grade)"].mean())
    out["second_sem_mean_gap"] = out["second_sem_non_dropout"] - out["second_sem_dropout"]
    sec_dropout = tmp.loc[tmp["Dropout_flag"] == 1, "Curricular units 2nd sem (grade)"].dropna()
    sec_non_dropout = tmp.loc[tmp["Dropout_flag"] == 0, "Curricular units 2nd sem (grade)"].dropna()
    out["second_sem_dropout_zero_rate"] = float((sec_dropout == 0).mean()) if len(sec_dropout) else float("nan")
    out["second_sem_non_dropout_zero_rate"] = float((sec_non_dropout == 0).mean()) if len(sec_non_dropout) else float("nan")
    out["second_sem_dropout_ge10_rate"] = float((sec_dropout >= 10).mean()) if len(sec_dropout) else float("nan")
    out["second_sem_dropout_ge12_rate"] = float((sec_dropout >= 12).mean()) if len(sec_dropout) else float("nan")

    q = pd.qcut(
        tmp["Curricular units 1st sem (grade)"],
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )
    q_rate = tmp.groupby(q, observed=True)["Dropout_flag"].mean()
    out["dropout_q1"] = float(q_rate.get("Q1", np.nan))
    out["dropout_q4"] = float(q_rate.get("Q4", np.nan))

    # Boxplot-style distribution diagnostics (median, IQR, 1.5*IQR outliers)
    def add_box_stats(feature: str, prefix: str) -> None:
        for cls, name in [(1, "dropout"), (0, "non_dropout")]:
            s = tmp.loc[tmp["Dropout_flag"] == cls, feature].dropna()
            if len(s) == 0:
                out[f"{prefix}_{name}_median"] = float("nan")
                out[f"{prefix}_{name}_iqr"] = float("nan")
                out[f"{prefix}_{name}_outlier_count"] = 0.0
                out[f"{prefix}_{name}_outlier_rate"] = float("nan")
                continue

            q1 = float(s.quantile(0.25))
            med = float(s.quantile(0.50))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            outliers = int(((s < low) | (s > high)).sum())

            out[f"{prefix}_{name}_median"] = med
            out[f"{prefix}_{name}_iqr"] = float(iqr)
            out[f"{prefix}_{name}_outlier_count"] = float(outliers)
            out[f"{prefix}_{name}_outlier_rate"] = float(outliers / len(s))

    add_box_stats("Admission grade", "admission")
    add_box_stats("Curricular units 1st sem (grade)", "first_sem")
    add_box_stats("Curricular units 2nd sem (grade)", "second_sem")
    return out


def build_model_tradeoff_text(comparison_df: pd.DataFrame) -> str:
    by_f1 = comparison_df.sort_values("f1", ascending=False).reset_index(drop=True)
    by_auc = comparison_df.sort_values("auc", ascending=False).reset_index(drop=True)
    by_precision = comparison_df.sort_values("precision", ascending=False).reset_index(drop=True)
    by_recall = comparison_df.sort_values("recall", ascending=False).reset_index(drop=True)

    best_f1 = by_f1.iloc[0]
    best_auc = by_auc.iloc[0]
    best_precision = by_precision.iloc[0]
    best_recall = by_recall.iloc[0]

    return (
        f"**Model comparison interpretation.** The strongest test-set F1 is from **{best_f1['model']}** "
        f"({best_f1['f1']:.3f}), while the highest AUC is from **{best_auc['model']}** ({best_auc['auc']:.3f}). "
        f"The best precision is **{best_precision['model']}** ({best_precision['precision']:.3f}), and the best recall is "
        f"**{best_recall['model']}** ({best_recall['recall']:.3f}). "
        "This pattern suggests a practical trade-off: tree ensembles and tuned classifiers provide stronger ranking power, "
        "while model choice should reflect intervention policy (catching more at-risk students vs. minimizing false alerts)."
    )


@st.cache_data
def load_metadata() -> Dict[str, Any]:
    with (META / "project_metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics() -> Dict[str, Any]:
    with (METRICS / "part2_metrics.json").open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_best_params() -> Dict[str, Any]:
    with (METRICS / "part2_best_params.json").open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_comparison_df() -> pd.DataFrame:
    return pd.read_csv(METRICS / "part2_model_comparison.csv")


@st.cache_data
def load_bonus_mlp_tuning_df() -> pd.DataFrame:
    path = METRICS / "bonus_mlp_tuning_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(
        columns=[
            "hidden_layers",
            "learning_rate",
            "dropout_rate",
            "val_f1",
            "epochs_trained",
            "best_val_loss",
        ]
    )


@st.cache_resource
def load_models() -> Dict[str, Any]:
    return {
        "logistic": joblib.load(MODELS / "logistic_pipeline.joblib"),
        "decision_tree": joblib.load(MODELS / "decision_tree_pipeline.joblib"),
        "random_forest": joblib.load(MODELS / "random_forest_pipeline.joblib"),
        "lightgbm": joblib.load(MODELS / "lightgbm_pipeline.joblib"),
        "mlp_keras": tf.keras.models.load_model(MODELS / "mlp_keras_model.keras"),
        "mlp_bundle": joblib.load(MODELS / "mlp_preprocess.joblib"),
    }


@st.cache_data
def compute_interactive_roc_points(
    _df: pd.DataFrame,
    _feature_names: List[str],
    _random_state: int = 42,
) -> pd.DataFrame:
    _, test_df = train_test_split(
        _df,
        test_size=0.30,
        stratify=_df["Dropout_flag"],
        random_state=_random_state,
    )
    X_test = test_df[_feature_names].copy()
    y_test = test_df["Dropout_flag"].astype(int).to_numpy()
    model_map = load_models()

    display = {
        "logistic": "Logistic Regression (Baseline)",
        "decision_tree": "Decision Tree (CART)",
        "random_forest": "Random Forest",
        "lightgbm": "LightGBM (Boosted Tree)",
        "mlp_keras": "MLP (Keras)",
    }

    rows: List[Dict[str, float]] = []
    for model_key in ["logistic", "decision_tree", "random_forest", "lightgbm", "mlp_keras"]:
        if model_key == "mlp_keras":
            bundle = model_map["mlp_bundle"]
            X_scaled = bundle["preprocess"].transform(X_test)
            y_prob = model_map["mlp_keras"].predict(X_scaled, verbose=0).ravel()
        else:
            y_prob = model_map[model_key].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        for f, t in zip(fpr, tpr):
            rows.append(
                {
                    "Model": display[model_key],
                    "False Positive Rate": float(f),
                    "True Positive Rate": float(t),
                }
            )

    return pd.DataFrame(rows)


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


def predict_with_model(
    model_name: str,
    row_df: pd.DataFrame,
    feature_names: List[str],
    models: Dict[str, Any],
    threshold: float = 0.5,
) -> Tuple[float, int]:
    if model_name == "mlp_keras":
        bundle = models["mlp_bundle"]
        X = bundle["preprocess"].transform(row_df[feature_names])
        prob = float(models["mlp_keras"].predict(X, verbose=0).ravel()[0])
    else:
        prob = float(models[model_name].predict_proba(row_df[feature_names])[:, 1][0])
    pred = int(prob >= float(threshold))
    return prob, pred


def make_custom_waterfall(
    tree_model_name: str,
    row_df: pd.DataFrame,
    feature_names: List[str],
    models: Dict[str, Any],
) -> plt.Figure:
    tree_pipe = models[tree_model_name]
    X_trans = tree_pipe.named_steps["preprocess"].transform(row_df[feature_names])
    X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

    tree_model = tree_pipe.named_steps["model"]
    explainer = shap.TreeExplainer(tree_model)
    shap_raw = explainer.shap_values(X_trans_df)
    shap_values, shap_base = extract_pos_class_shap(shap_raw, explainer.expected_value)

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=shap_base,
        data=X_trans_df.iloc[0].values,
        feature_names=feature_names,
    )

    fig = plt.figure(figsize=(8, 5.5))
    shap.plots.waterfall(explanation, show=False, max_display=12)
    plt.tight_layout()
    return fig


def compute_custom_shap_contributions(
    tree_model_name: str,
    row_df: pd.DataFrame,
    feature_names: List[str],
    models: Dict[str, Any],
    top_n: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tree_pipe = models[tree_model_name]
    X_trans = tree_pipe.named_steps["preprocess"].transform(row_df[feature_names])
    X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

    tree_model = tree_pipe.named_steps["model"]
    explainer = shap.TreeExplainer(tree_model)
    shap_raw = explainer.shap_values(X_trans_df)
    shap_values, _ = extract_pos_class_shap(shap_raw, explainer.expected_value)
    row_shap = np.array(shap_values[0], dtype=float)

    contrib_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Input Value": X_trans_df.iloc[0].values,
            "SHAP Value": row_shap,
        }
    )
    positive = contrib_df.sort_values("SHAP Value", ascending=False).head(top_n).copy()
    negative = contrib_df.sort_values("SHAP Value", ascending=True).head(top_n).copy()
    return positive, negative


def get_decision_tree_level_notes(tree_pipe: Any, feature_names: List[str], max_depth: int = 2) -> List[str]:
    try:
        model = tree_pipe.named_steps["model"]
        tree = model.tree_
    except Exception:
        return []

    classes = list(getattr(model, "classes_", [0, 1]))
    dropout_idx = classes.index(1) if 1 in classes else min(1, len(classes) - 1)
    leaf_flag = -2

    notes: List[str] = []
    queue: List[Tuple[int, int, str]] = [(0, 0, "Root node")]

    while queue:
        node_id, depth, label = queue.pop(0)
        values = np.array(tree.value[node_id][0], dtype=float)
        total = float(values.sum())
        dropout_share = float(values[dropout_idx] / total) if total > 0 else float("nan")
        n_samples = int(tree.n_node_samples[node_id])

        feat_idx = int(tree.feature[node_id])
        if feat_idx != leaf_flag:
            feat_name = feature_names[feat_idx] if 0 <= feat_idx < len(feature_names) else f"feature_{feat_idx}"
            threshold = float(tree.threshold[node_id])
            notes.append(
                f"{label} (depth {depth}): split on `{feat_name} <= {threshold:.2f}` "
                f"(node dropout share: {dropout_share:.1%}, samples: {n_samples})."
            )
            if depth < max_depth:
                queue.append((int(tree.children_left[node_id]), depth + 1, f"Left child from depth {depth}"))
                queue.append((int(tree.children_right[node_id]), depth + 1, f"Right child from depth {depth}"))
        else:
            pred_idx = int(np.argmax(values)) if len(values) else 0
            pred_class = classes[pred_idx] if pred_idx < len(classes) else pred_idx
            notes.append(
                f"{label} (depth {depth}) is a leaf: predicts `{pred_class}` "
                f"(node dropout share: {dropout_share:.1%}, samples: {n_samples})."
            )

    return notes


def get_decision_tree_path_summary(tree_pipe: Any, feature_names: List[str]) -> Dict[str, Any]:
    try:
        model = tree_pipe.named_steps["model"]
        tree = model.tree_
    except Exception:
        return {}

    classes = list(getattr(model, "classes_", [0, 1]))
    dropout_idx = classes.index(1) if 1 in classes else min(1, len(classes) - 1)
    leaf_flag = -2

    leaf_rows: List[Dict[str, Any]] = []

    def walk(node_id: int, rules: List[str]) -> None:
        feat_idx = int(tree.feature[node_id])
        if feat_idx == leaf_flag:
            values = np.array(tree.value[node_id][0], dtype=float)
            total = float(values.sum())
            dropout_share = float(values[dropout_idx] / total) if total > 0 else float("nan")
            leaf_rows.append(
                {
                    "path": " and ".join(rules) if rules else "(no split)",
                    "dropout_share": dropout_share,
                    "samples": int(tree.n_node_samples[node_id]),
                }
            )
            return

        feat_name = feature_names[feat_idx] if 0 <= feat_idx < len(feature_names) else f"feature_{feat_idx}"
        threshold = float(tree.threshold[node_id])
        walk(int(tree.children_left[node_id]), rules + [f"{feat_name} <= {threshold:.2f}"])
        walk(int(tree.children_right[node_id]), rules + [f"{feat_name} > {threshold:.2f}"])

    walk(0, [])

    if not leaf_rows:
        return {}

    leaf_df = pd.DataFrame(leaf_rows).sort_values("samples", ascending=False)
    high = leaf_df.sort_values(["dropout_share", "samples"], ascending=[False, False]).iloc[0].to_dict()
    low = leaf_df.sort_values(["dropout_share", "samples"], ascending=[True, False]).iloc[0].to_dict()

    root_values = np.array(tree.value[0][0], dtype=float)
    root_total = float(root_values.sum())
    root_share = float(root_values[dropout_idx] / root_total) if root_total > 0 else float("nan")

    return {
        "root_dropout_share": root_share,
        "high_risk": high,
        "low_risk": low,
    }


@st.cache_data
def compute_shap_interpretation(
    _df: pd.DataFrame,
    _feature_names: List[str],
    _best_tree_model: str,
    _random_state: int = 42,
) -> List[Dict[str, Any]]:
    model_map = load_models()
    tree_pipe = model_map[_best_tree_model]
    _, test_df = train_test_split(
        _df,
        test_size=0.30,
        stratify=_df["Dropout_flag"],
        random_state=_random_state,
    )
    X = test_df[_feature_names].copy()
    X_trans = tree_pipe.named_steps["preprocess"].transform(X)
    X_trans_df = pd.DataFrame(X_trans, columns=_feature_names)

    explainer = shap.TreeExplainer(tree_pipe.named_steps["model"])
    shap_raw = explainer.shap_values(X_trans_df)
    shap_values, _ = extract_pos_class_shap(shap_raw, explainer.expected_value)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:5]
    output: List[Dict[str, Any]] = []

    for idx in top_idx:
        feat = _feature_names[int(idx)]
        corr = np.corrcoef(X_trans_df[feat].to_numpy(), shap_values[:, int(idx)])[0, 1]
        if np.isnan(corr):
            direction = "direction is weak or non-linear in aggregate"
        elif corr > 0:
            direction = "higher values tend to increase dropout risk"
        else:
            direction = "higher values tend to decrease dropout risk"
        output.append(
            {
                "feature": feat,
                "mean_abs_shap": float(mean_abs[int(idx)]),
                "direction": direction,
            }
        )
    return output


@st.cache_data
def compute_reference_waterfall_details(
    _df: pd.DataFrame,
    _feature_names: List[str],
    _best_tree_model: str,
    _random_state: int = 42,
) -> Dict[str, Any]:
    model_map = load_models()
    tree_pipe = model_map[_best_tree_model]

    _, test_df = train_test_split(
        _df,
        test_size=0.30,
        stratify=_df["Dropout_flag"],
        random_state=_random_state,
    )
    X_raw = test_df[_feature_names].copy()
    probs = tree_pipe.predict_proba(X_raw)[:, 1]
    ref_pos = int(np.argmax(probs))
    ref_idx_original = int(test_df.index[ref_pos])

    X_trans = tree_pipe.named_steps["preprocess"].transform(X_raw)
    X_trans_df = pd.DataFrame(X_trans, columns=_feature_names)

    tree_model = tree_pipe.named_steps["model"]
    explainer = shap.TreeExplainer(tree_model)
    shap_raw = explainer.shap_values(X_trans_df.iloc[[ref_pos]])
    shap_values, _ = extract_pos_class_shap(shap_raw, explainer.expected_value)
    row_shap = np.array(shap_values[0], dtype=float)

    detail_df = pd.DataFrame(
        {
            "Feature": _feature_names,
            "Feature Value": X_trans_df.iloc[ref_pos].values,
            "SHAP Value": row_shap,
        }
    )
    top_risk = detail_df.sort_values("SHAP Value", ascending=False).head(3).copy()
    top_protective = detail_df.sort_values("SHAP Value", ascending=True).head(3).copy()

    return {
        "reference_index": ref_idx_original,
        "reference_test_position": ref_pos,
        "reference_probability": float(probs[ref_pos]),
        "top_risk": top_risk.to_dict(orient="records"),
        "top_protective": top_protective.to_dict(orient="records"),
    }


meta = load_metadata()
metrics = load_metrics()
best_params = load_best_params()
comparison_df = load_comparison_df()
bonus_mlp_tuning_df = load_bonus_mlp_tuning_df()
df = load_data()
models = load_models()
missing_total = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

feature_names = meta["feature_selection"]["final_features"]
roc_points_df = compute_interactive_roc_points(
    df,
    feature_names,
    int(meta.get("random_state", 42)),
)
feature_selection_meta = meta.get("feature_selection", {})
selected_18_rechecked = feature_selection_meta.get("selected_18_rechecked", [])
selected_10_candidate = feature_selection_meta.get("selected_10", [])
cv_f1_18 = float(feature_selection_meta.get("cv_f1_18", np.nan))
cv_f1_10 = float(feature_selection_meta.get("cv_f1_10", np.nan))
feature_choice_logic = meta.get("final_model_choice_logic", "")
corr_columns = get_full_corr_columns(df)
corr_full = df[corr_columns].corr()
focus_corr_columns = get_top_target_corr_columns(corr_full, target_col="Dropout_flag", top_k=10)
manual_input_features = meta.get("manual_input_features", [])
feature_ranges = meta["feature_ranges"]
feature_means = meta["feature_means"]
best_tree_model = meta["best_tree_model_for_shap"]
captions = meta["captions"]
top_corr_a, top_corr_b, top_corr_val = strongest_corr_pair(df, corr_columns)
corr_first_second_grade = get_corr_value(corr_full, "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)")
corr_first_second_approved = get_corr_value(corr_full, "Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)")
corr_age_app_mode = get_corr_value(corr_full, "Age at enrollment", "Application mode")
corr_target_second_grade = get_corr_value(corr_full, "Curricular units 2nd sem (grade)", "Dropout_flag")
corr_target_second_approved = get_corr_value(corr_full, "Curricular units 2nd sem (approved)", "Dropout_flag")
corr_target_tuition = get_corr_value(corr_full, "Tuition fees up to date", "Dropout_flag")
corr_target_debtor = get_corr_value(corr_full, "Debtor", "Dropout_flag")
corr_target_scholarship = get_corr_value(corr_full, "Scholarship holder", "Dropout_flag")
corr_target_age = get_corr_value(corr_full, "Age at enrollment", "Dropout_flag")
target_corr_abs = (
    corr_full["Dropout_flag"].drop(labels=["Dropout_flag"], errors="ignore").abs().sort_values(ascending=False)
    if "Dropout_flag" in corr_full.columns
    else pd.Series(dtype=float)
)
top_target_feature = str(target_corr_abs.index[0]) if len(target_corr_abs) else "N/A"
top_target_corr = float(corr_full.loc[top_target_feature, "Dropout_flag"]) if len(target_corr_abs) else float("nan")
shap_interpretation = compute_shap_interpretation(
    df,
    feature_names,
    best_tree_model,
    int(meta.get("random_state", 42)),
)
reference_waterfall = compute_reference_waterfall_details(
    df,
    feature_names,
    best_tree_model,
    int(meta.get("random_state", 42)),
)
eda_highlights = compute_eda_highlights(df)
tradeoff_text = build_model_tradeoff_text(comparison_df)
original_counts = meta["dataset_stats"].get("target_counts_original", {})
total_n = int(len(df))
train_n = int(total_n * 0.70)
test_n = total_n - train_n
exec_model_name_map = {
    "logistic": "Logistic Regression",
    "decision_tree": "Decision Tree (CART)",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM (Boosted Tree)",
    "mlp_keras": "MLP (Keras)",
}
exec_best_f1_row = comparison_df.sort_values("f1", ascending=False).iloc[0]
exec_best_auc_row = comparison_df.sort_values("auc", ascending=False).iloc[0]
exec_best_f1_model = exec_model_name_map.get(str(exec_best_f1_row["model"]), str(exec_best_f1_row["model"]))
exec_best_auc_model = exec_model_name_map.get(str(exec_best_auc_row["model"]), str(exec_best_auc_row["model"]))

st.title("Student Dropout Risk Modeling for Early Intervention")
st.caption(
    "Dataset: UCI Machine Learning Repository - Predict Students' Dropout and Academic Success"
)


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ]
)


with tab1:
    st.subheader("About This Project")
    st.markdown(
        "This project asks a deceptively simple question: can we look at a student's background and early academic record "
        "and predict, before it's too late, whether they're heading toward dropping out? "
        "The data I'm working with comes from the UCI Machine Learning Repository, a publicly available dataset "
        "collected at a higher education institution in Portugal. "
        "It covers **4,424 students** across multiple degree programs, tracking each person from their enrollment paperwork "
        "through their performance in the first two semesters. "
        "Source: [UCI Dataset Page](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)"
    )
    st.markdown(
        "The dataset has **36 features** per student, a mix of numeric (real and integer) and categorically-coded variables. "
        "Some of those are things you'd expect: age, gender, which program they enrolled in, their admission grades. "
        "Others are less obvious but turn out to be quite predictive: whether they had a scholarship, "
        "whether they were behind on tuition, whether they had debt, "
        "and how their parents' education level and occupation compared to other students. "
        "Academic performance in the first semester is also recorded: approved units, grades, and whether they even attempted evaluations. "
        "The original data labels each student with one of three outcomes: **Dropout**, **Enrolled** (still in progress), or **Graduate**. "
        "For this project, I collapsed that into a simpler yes/no question: did this person drop out or not?"
    )
    if original_counts:
        st.markdown(
            f"Of the {total_n} students in the dataset, "
            f"**{original_counts.get('Dropout', 'N/A')} dropped out**, "
            f"{original_counts.get('Enrolled', 'N/A')} were still enrolled at the time data was collected, "
            f"and {original_counts.get('Graduate', 'N/A')} had graduated. "
            f"That puts the overall dropout rate at roughly **{eda_highlights['dropout_rate']:.1%}**, "
            "meaning about one in three students who started never finished."
        )
    st.markdown(
        f"Data quality checks found **{missing_total} missing values** and **{duplicate_rows} duplicate rows** in this dataset, "
        "so the records are clean and ready to use without any major preprocessing issues. "
        "The final binary prediction target is: **Dropout = 1** vs **Non-dropout (Enrolled + Graduate) = 0**."
    )

    st.subheader("Why This Problem Matters")
    st.markdown(
        "Dropout is one of those outcomes that's easy to observe after the fact and surprisingly hard to catch in advance. "
        "By the time a student officially withdraws, they've usually been struggling for weeks or months, "
        "missing classes, falling behind on assignments, quietly disengaging. "
        "If a university could identify those students early, even just a few weeks into a semester, "
        "they could reach out and offer support before things spiral. "
        "That's the core promise of a dropout prediction model."
    )
    st.markdown(
        "The impact goes well beyond a single student. Universities have limited advising staff, limited financial aid budgets, "
        "and limited capacity for one-on-one support. "
        "A good prediction model helps those resources go to the right people at the right time, "
        "instead of spreading thinly across everyone or, even worse, going to the students who are most vocal rather than most at risk. "
        "Research consistently shows that retaining a current student is far less expensive than recruiting a new one, "
        "so early intervention also makes sense from a pure resource-allocation standpoint."
    )
    st.markdown(
        "There's also an equity angle. Dropout risk isn't distributed evenly. "
        "Students who are older, financially stressed, or from less-advantaged backgrounds tend to drop out at higher rates. "
        "A model that surfaces those patterns explicitly can help institutions direct support more fairly, "
        "rather than relying on which students are most visible or most proactive about asking for help."
    )

    st.subheader("What I Did and What I Found")
    st.markdown(
        "The approach here is a full machine learning pipeline, from data exploration through model deployment. "
        "I trained five different classification models on the same dataset: a logistic regression (baseline), "
        "a single decision tree, a random forest, a gradient-boosted tree (LightGBM), and a small neural network. "
        "Each model was tuned using cross-validation on the training data only, then evaluated on a held-out test set "
        "that none of them saw during training. "
        "I used F1 score as the main criterion for comparing models, since it balances catching real dropout cases "
        "against generating too many false alarms."
    )
    st.markdown(
        f"The clearest finding in the data is that **first-semester academic performance is by far the strongest early signal**. "
        f"Students who dropped out averaged a grade of **{eda_highlights['first_sem_dropout']:.2f}** in their first semester, "
        f"compared to **{eda_highlights['first_sem_non_dropout']:.2f}** among students who stayed. "
        "That gap shows up consistently across every analysis I ran, and it suggests that the window for intervention "
        "is really the first few months of enrollment, not later when students are already disengaged. "
        "Financial signals also played a meaningful role: students carrying debt or behind on tuition were substantially more likely to drop out "
        "than peers without those pressures."
    )
    st.markdown(
        f"The best-performing model overall was **{exec_best_f1_model}**, "
        f"with a test-set F1 of **{float(exec_best_f1_row['f1']):.3f}**, "
        f"accuracy of **{float(exec_best_f1_row['accuracy']):.3f}**, "
        f"and AUC of **{float(exec_best_f1_row['auc']):.3f}**. "
        "The AUC in particular is worth unpacking for non-technical readers: it measures how well the model can rank students by risk level. "
        "A score of 1.0 would mean perfect ranking; 0.5 would be no better than random. "
        f"Getting above 0.90 means the model is genuinely good at separating high-risk students from low-risk ones across the whole population."
    )
    if str(exec_best_f1_row["model"]) != str(exec_best_auc_row["model"]):
        st.markdown(
            f"One nuance worth noting: **{exec_best_auc_model}** had the highest AUC at **{float(exec_best_auc_row['auc']):.3f}**, "
            f"which means it ranked students by risk slightly better across the board. "
            f"**{exec_best_f1_model}** had the better precision-recall balance at the standard 50% probability cutoff. "
            "Depending on what the institution wants to optimize for (a tight prioritized list or a broader safety net), "
            "either could be the right operational choice."
        )

    st.subheader("How to Use These Results")
    st.markdown(
        "What this model produces for each student is a dropout risk score, a number between 0 and 1 "
        "that reflects how likely that person is to drop out based on their profile. "
        "Students scoring above a chosen threshold get flagged for outreach; students below it don't. "
        "Crucially, the model also explains each individual prediction through a breakdown of which specific factors drove the score up or down. "
        "That's what the SHAP analysis in the last tab shows: instead of a black-box result, "
        "you get something like 'this student's risk is elevated mainly because of low first-semester grades and overdue tuition fees.'"
    )
    st.markdown(
        "In a practical workflow, I'd suggest running predictions at the start of each semester, "
        "sorting students into a few rough risk tiers, and routing them to the right kind of support based on what's actually driving their score. "
        "Academic struggles call for tutoring, early faculty alerts, and mandatory check-ins. "
        "Financial stress calls for payment plan counseling and emergency aid. "
        "When both show up together, that's when coordinated outreach from both advising and financial aid is most important. "
        "The model is meant to help prioritize who gets contacted first, "
        "but the decision about what kind of help to offer still belongs to the people doing the advising. "
        "And predictions should always be used to allocate support, not to exclude or penalize students."
    )

    ds = meta["dataset_stats"]
    st.subheader("Quick Dataset Stats")
    original_total_columns = int(ds["columns"]) - 2
    original_predictor_columns = max(0, original_total_columns - 1)
    st.dataframe(
        pd.DataFrame(
            {
                "Students": [ds["rows"]],
                "Original UCI Predictors": [original_predictor_columns],
                "Original UCI Target Columns": [1],
                "Original UCI Total Columns": [original_total_columns],
                "Dropout Rate": [round(ds["target_ratio_binary"]["1"], 4)],
                "Non-dropout Rate": [round(ds["target_ratio_binary"]["0"], 4)],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Original UCI dataset has 36 features plus 1 target column (37 total). "
        "The pipeline adds 2 derived columns internally (Dropout_flag, Target_label), so the working dataframe is 39 columns."
    )


with tab2:
    st.header("Part 1: Descriptive Analytics")

    st.subheader("1.2 Target Distribution")

    st.image(str(FIGURES / "part1_target_distribution.png"), use_container_width=True)
    st.caption(captions["target_distribution"])
    target_counts_df = (
        df["Dropout_flag"]
        .value_counts()
        .rename_axis("Dropout_flag")
        .reset_index(name="Count")
        .sort_values("Dropout_flag")
        .reset_index(drop=True)
    )
    target_counts_df["Outcome"] = target_counts_df["Dropout_flag"].map({0: "Non-dropout (0)", 1: "Dropout (1)"})
    target_counts_df["Rate"] = target_counts_df["Count"] / float(target_counts_df["Count"].sum())
    st.markdown("**Interactive view (hover for exact counts and ratios):**")
    target_chart = (
        alt.Chart(target_counts_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Outcome:N", title="Class"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color("Outcome:N", legend=None, scale=alt.Scale(range=["#4C72B0", "#C44E52"])),
            tooltip=[
                alt.Tooltip("Outcome:N"),
                alt.Tooltip("Count:Q", format=","),
                alt.Tooltip("Rate:Q", format=".1%"),
            ],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(target_chart, use_container_width=True)
    st.markdown(
        f"Observed class mix: **Dropout = {eda_highlights['dropout_rate']:.1%}** and **Non-dropout = {(1-eda_highlights['dropout_rate']):.1%}**. "
        "This confirms that F1/AUC are more reliable than accuracy alone for model selection."
    )
    st.markdown(
        "This is a **classification** target, so a class-frequency bar chart is the correct plot here. "
        "Histogram/KDE plots are used for continuous regression targets."
    )
    st.markdown(
        "The target distribution is skewed toward the non-dropout class. "
        "For a categorical target, outliers are not interpreted the same way as in continuous targets; the main issue is class imbalance."
    )
    if original_counts:
        enrolled_n = int(original_counts.get("Enrolled", 0))
        graduate_n = int(original_counts.get("Graduate", 0))
        dropout_n = int(original_counts.get("Dropout", 0))
        non_dropout_n = enrolled_n + graduate_n
        st.markdown(
            f"Class definition for this binary chart: **Dropout (1) = original Dropout ({dropout_n})**; "
            f"**Non-dropout (0) = original Enrolled + Graduate ({enrolled_n} + {graduate_n} = {non_dropout_n})**."
        )
    st.markdown(
        "Handling this imbalance carefully mattered throughout the project. "
        "I used a stratified train/test split to preserve the same class ratio in both sets, "
        "applied class weights during training so the models are penalized more heavily for missing actual dropout cases, "
        "evaluated primarily on F1 and AUC rather than raw accuracy, "
        "and reviewed the precision/recall tradeoff for each model to understand what kind of error each one makes."
    )

    st.subheader("1.3 Feature Distributions and Relationships")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(str(FIGURES / "part1_admission_grade_boxplot.png"), use_container_width=True)
        st.caption(captions["admission_grade_boxplot"])
        st.markdown(
            f"Mean admission grade is **{eda_highlights['admission_dropout']:.2f}** for dropout students "
            f"vs **{eda_highlights['admission_non_dropout']:.2f}** for non-dropout students. "
            "This is not the largest gap in the dataset, but it is directionally consistent with risk."
        )
        st.markdown(
            f"Distribution detail: median is **{eda_highlights['admission_dropout_median']:.2f}** (dropout) vs "
            f"**{eda_highlights['admission_non_dropout_median']:.2f}** (non-dropout), and the middle-50% range is "
            f"**{eda_highlights['admission_dropout_iqr']:.2f}** vs **{eda_highlights['admission_non_dropout_iqr']:.2f}**. "
            f"Students with unusually high/low values (outliers) account for **{eda_highlights['admission_dropout_outlier_rate']:.1%}** "
            f"in the dropout group and **{eda_highlights['admission_non_dropout_outlier_rate']:.1%}** in the non-dropout group. "
            "In plain terms, the two groups overlap, but the center of the dropout group is still shifted lower."
        )
    with col_b:
        st.image(str(FIGURES / "part1_first_sem_grade_boxplot.png"), use_container_width=True)
        st.caption(captions["first_sem_grade_boxplot"])
        st.markdown(
            f"Mean first-semester grade is **{eda_highlights['first_sem_dropout']:.2f}** for dropout students "
            f"and **{eda_highlights['first_sem_non_dropout']:.2f}** for non-dropout students. "
            "This large separation supports using early academic performance as a primary intervention trigger."
        )
        st.markdown(
            f"Distribution detail: median is **{eda_highlights['first_sem_dropout_median']:.2f}** (dropout) vs "
            f"**{eda_highlights['first_sem_non_dropout_median']:.2f}** (non-dropout), with middle-50% range "
            f"**{eda_highlights['first_sem_dropout_iqr']:.2f}** vs **{eda_highlights['first_sem_non_dropout_iqr']:.2f}**. "
            f"The share of unusual values is **{eda_highlights['first_sem_dropout_outlier_rate']:.1%}** in the dropout group "
            f"and **{eda_highlights['first_sem_non_dropout_outlier_rate']:.1%}** in the non-dropout group. "
            "The larger center gap than in admission scores suggests first-semester performance is a much sharper early-separation signal."
        )

    fig_second_sem_violin = make_second_sem_violin_figure(df)
    st.pyplot(fig_second_sem_violin, clear_figure=True)
    plt.close(fig_second_sem_violin)
    st.caption(
        "Violin plot of second-semester grades by outcome. "
        "Each violin shows the full distribution shape for one group, the inner box shows median/IQR, and the black dot marks the mean."
    )
    st.markdown(
        f"Second-semester results separate the groups very clearly: the mean is **{eda_highlights['second_sem_dropout']:.2f}** in the dropout group "
        f"versus **{eda_highlights['second_sem_non_dropout']:.2f}** in the non-dropout group "
        f"(a gap of **{eda_highlights['second_sem_mean_gap']:.2f}** points)."
    )
    st.markdown(
        f"The dropout violin is much wider near zero because **{eda_highlights['second_sem_dropout_zero_rate']:.1%}** of dropout students have a second-semester grade of 0, "
        f"compared with only **{eda_highlights['second_sem_non_dropout_zero_rate']:.1%}** in the non-dropout group. "
        f"This matches the center statistics (median **{eda_highlights['second_sem_dropout_median']:.2f}** vs **{eda_highlights['second_sem_non_dropout_median']:.2f}**)."
    )
    st.markdown(
        "How to read violin width: a wider section means **more students are concentrated at that score range**. "
        "So the wide lower part in the dropout violin means many dropout students are clustered at very low/zero grades, "
        "while the non-dropout violin is widest around higher grades."
    )
    st.markdown(
        f"The upper-side shoulder is also meaningful: the dropout violin has a visible density concentration in the 10-14 range. "
        f"In this dataset, about **{eda_highlights['second_sem_dropout_ge10_rate']:.1%}** of dropout students are at **10+**, "
        f"and **{eda_highlights['second_sem_dropout_ge12_rate']:.1%}** are at **12+**. "
        "That means dropout risk is not explained by grades alone; a meaningful subgroup has decent grades but likely faces other pressures "
        "(financial strain, attendance/engagement, or delayed administrative issues)."
    )
    st.markdown(
        "Practical takeaway: this is not a small shift. "
        "If a student reaches very low or zero second-semester performance, treat it as a high-priority intervention trigger "
        "(advisor outreach + academic support + financial check) rather than waiting for later outcomes."
    )

    fig_grade_band = make_grade_band_dropout_figure(df)
    st.pyplot(fig_grade_band, clear_figure=True)
    plt.close(fig_grade_band)
    st.caption(captions["dropout_by_grade_quartile"])
    st.markdown(
        f"Dropout risk by grade band is highly nonlinear: **Lowest 25% = {eda_highlights['dropout_q1']:.1%}** vs **Highest 25% = {eda_highlights['dropout_q4']:.1%}**. "
        "In practical terms, students in the lowest performance quartile should be prioritized for early support."
    )
    st.markdown(
        "How to read this chart: "
        "1) students are sorted by first-semester grade, "
        "2) then split into four equal-size grade bands, and "
        "3) each bar shows the dropout percentage inside that group. "
        "So this is a risk-by-group view, not raw grade values on the y-axis."
    )
    st.markdown(
        "Important: these are **grade percentile bands**, not semester quarters."
    )
    st.markdown(
        "Actionable insight: because the **lowest 25% grade band** has the highest dropout rate, this group should receive first-priority support "
        "(early advising outreach, tutoring, and financial-risk check-ins) before risk compounds."
    )
    st.markdown(
        f"Financial signals show a similar risk pattern: debtors have **{eda_highlights['dropout_debtor_1']:.1%}** dropout "
        f"vs **{eda_highlights['dropout_debtor_0']:.1%}** for non-debtors; scholarship holders have **{eda_highlights['dropout_scholar_1']:.1%}** "
        f"vs **{eda_highlights['dropout_scholar_0']:.1%}** for non-holders; and students with tuition fees **not up to date** show "
        f"**{eda_highlights['dropout_tuition_0']:.1%}** dropout vs **{eda_highlights['dropout_tuition_1']:.1%}** when fees are up to date. "
        "These financial differences are visualized directly in the charts right below."
    )

    col_c, col_d, col_e = st.columns(3)
    with col_c:
        fig_debtor = make_dropout_rate_figure(
            df=df,
            feature="Debtor",
            labels={0: "Non-debtor", 1: "Debtor"},
            title="Dropout Rate by Debtor Status",
        )
        st.pyplot(fig_debtor, clear_figure=True)
        plt.close(fig_debtor)
        st.caption(
            "Debtor status is strongly associated with dropout risk. "
            "The debtor group shows a materially higher dropout rate, indicating that financial pressure is a practical early-warning signal for intervention."
        )
    with col_d:
        fig_scholar = make_dropout_rate_figure(
            df=df,
            feature="Scholarship holder",
            labels={0: "No Scholarship", 1: "Scholarship"},
            title="Dropout Rate by Scholarship Status",
        )
        st.pyplot(fig_scholar, clear_figure=True)
        plt.close(fig_scholar)
        st.caption(
            "Scholarship support is associated with lower dropout risk. "
            "Students with scholarships show a substantially lower dropout rate, consistent with financial support acting as a retention buffer."
        )
    with col_e:
        fig_tuition = make_dropout_rate_figure(
            df=df,
            feature="Tuition fees up to date",
            labels={0: "Fees Not Up to Date", 1: "Fees Up to Date"},
            title="Dropout Rate by Tuition Status",
        )
        st.pyplot(fig_tuition, clear_figure=True)
        plt.close(fig_tuition)
        st.caption(
            "Tuition status is a strong risk separator. "
            "Students with unpaid or overdue tuition show a much higher dropout rate, so payment-status monitoring can be used as an operational early-warning signal."
        )

    st.subheader("1.4 Correlation Heatmap")
    fig_corr_focus = make_focused_correlation_heatmap_figure(df, focus_corr_columns)
    st.pyplot(fig_corr_focus, clear_figure=True)
    plt.close(fig_corr_focus)
    st.caption(
        "Focused heatmap using the top target-linked variables (highest |correlation with `Dropout_flag`) plus the target. "
        "This view improves readability for interpretation."
    )
    corr_focus_df = df[focus_corr_columns].corr().reset_index().rename(columns={"index": "Feature_Y"})
    corr_focus_long = corr_focus_df.melt(id_vars="Feature_Y", var_name="Feature_X", value_name="Correlation")
    st.markdown("**Interactive focused heatmap (hover to inspect each pair):**")
    corr_chart = (
        alt.Chart(corr_focus_long)
        .mark_rect()
        .encode(
            x=alt.X("Feature_X:N", title="", sort=focus_corr_columns, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("Feature_Y:N", title="", sort=focus_corr_columns),
            color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
            tooltip=[
                alt.Tooltip("Feature_Y:N", title="Row"),
                alt.Tooltip("Feature_X:N", title="Column"),
                alt.Tooltip("Correlation:Q", format=".3f"),
            ],
        )
        .properties(height=420)
        .interactive()
    )
    st.altair_chart(corr_chart, use_container_width=True)
    if not np.isnan(top_target_corr):
        st.markdown(
            f"Strongest target-linked variable in the full numeric set: **{top_target_feature}** "
            f"with `Dropout_flag` (**r = {top_target_corr:.3f}**)."
        )
    st.markdown(
        "Interpretation note: correlations for integer-coded categorical variables should be used as directional references, "
        "not as linear-effect magnitudes."
    )
    with st.expander("Show full correlation heatmap (all numeric columns)"):
        fig_corr_full = make_full_correlation_heatmap_figure(df, corr_columns)
        st.pyplot(fig_corr_full, clear_figure=True)
        plt.close(fig_corr_full)
        if not np.isnan(top_corr_val):
            st.markdown(
                f"Strongest absolute correlation in the full numeric set: **{top_corr_a}** vs **{top_corr_b}** "
                f"with **|r| = {top_corr_val:.3f}**."
            )
    st.markdown("**Reading the heatmap:**")
    st.markdown(
        f"Academic performance carries through across semesters: first-semester and second-semester grades are tightly linked "
        f"(**r = {corr_first_second_grade:.3f}**), and the same pattern shows up for approved units (**r = {corr_first_second_approved:.3f}**). "
        "That persistence is actually useful for intervention: a student struggling in semester one is likely to struggle again in semester two, "
        "so acting early has real leverage. "
        f"Application mode and age at enrollment also correlate moderately (**r = {corr_age_app_mode:.3f}**), "
        "which makes sense since older students often come through different admission pathways. "
        "This overlap is worth keeping in mind when selecting features, since the two variables might be partially redundant."
    )
    st.markdown(
        f"Looking directly at the dropout target, second-semester performance has one of the strongest associations "
        f"(grade r = {corr_target_second_grade:.3f}, approved-units r = {corr_target_second_approved:.3f}). "
        "The negative sign is what you'd expect: higher grades mean lower dropout risk. "
        f"Financial variables follow the same pattern: tuition up-to-date is negatively related to dropout (**r = {corr_target_tuition:.3f}**), "
        f"debtor status is positively related (**r = {corr_target_debtor:.3f}**), "
        f"and scholarship status is protective (**r = {corr_target_scholarship:.3f}**). "
        f"Age at enrollment shows a small positive association with dropout risk (**r = {corr_target_age:.3f}**), "
        "which may reflect the additional pressures older students often face, including work, family obligations, and less scheduling flexibility."
    )
    st.markdown(
        "For modeling, the main takeaway is that academic-progress and financial-status variables carry the clearest signal, "
        "while features with very high inter-correlations need to be filtered carefully to avoid multicollinearity. "
        "Correlation alone is only pairwise association, so the tree-based models and SHAP analysis in later tabs "
        "fill in the picture by capturing how multiple features interact together."
    )


with tab3:
    st.header("Part 2: Predictive Analytics")
    st.markdown(
        "This tab surfaces all Part 2 outputs in one place: model metrics, tuning results, model-comparison charts, "
        "best hyperparameters, and ROC curves for every classifier."
    )

    def one_row_metrics(model_key: str, label: str) -> pd.DataFrame:
        m = metrics.get(model_key, {})
        return pd.DataFrame(
            [
                {
                    "Model": label,
                    "Accuracy": m.get("accuracy", np.nan),
                    "Precision": m.get("precision", np.nan),
                    "Recall": m.get("recall", np.nan),
                    "F1": m.get("f1", np.nan),
                    "AUC-ROC": m.get("auc", np.nan),
                }
            ]
        )

    baseline_f1 = float(metrics.get("logistic", {}).get("f1", np.nan))
    baseline_auc = float(metrics.get("logistic", {}).get("auc", np.nan))
    baseline_precision = float(metrics.get("logistic", {}).get("precision", np.nan))
    baseline_recall = float(metrics.get("logistic", {}).get("recall", np.nan))
    baseline_accuracy = float(metrics.get("logistic", {}).get("accuracy", np.nan))
    dt_metrics = metrics.get("decision_tree", {})
    dt_accuracy = float(dt_metrics.get("accuracy", np.nan))
    dt_precision = float(dt_metrics.get("precision", np.nan))
    dt_recall = float(dt_metrics.get("recall", np.nan))
    dt_f1 = float(dt_metrics.get("f1", np.nan))
    dt_auc = float(dt_metrics.get("auc", np.nan))
    dt_params = best_params.get("decision_tree", {})
    dt_level_notes = get_decision_tree_level_notes(models.get("decision_tree"), feature_names, max_depth=2)
    dt_path_summary = get_decision_tree_path_summary(models.get("decision_tree"), feature_names)
    rf_metrics = metrics.get("random_forest", {})
    rf_accuracy = float(rf_metrics.get("accuracy", np.nan))
    rf_precision = float(rf_metrics.get("precision", np.nan))
    rf_recall = float(rf_metrics.get("recall", np.nan))
    rf_f1 = float(rf_metrics.get("f1", np.nan))
    rf_auc = float(rf_metrics.get("auc", np.nan))
    rf_params = best_params.get("random_forest", {})
    lgbm_metrics = metrics.get("lightgbm", {})
    lgbm_accuracy = float(lgbm_metrics.get("accuracy", np.nan))
    lgbm_precision = float(lgbm_metrics.get("precision", np.nan))
    lgbm_recall = float(lgbm_metrics.get("recall", np.nan))
    lgbm_f1 = float(lgbm_metrics.get("f1", np.nan))
    lgbm_auc = float(lgbm_metrics.get("auc", np.nan))
    lgbm_params = best_params.get("lightgbm", {})
    mlp_metrics = metrics.get("mlp_keras", {})
    mlp_accuracy = float(mlp_metrics.get("accuracy", np.nan))
    mlp_precision = float(mlp_metrics.get("precision", np.nan))
    mlp_recall = float(mlp_metrics.get("recall", np.nan))
    mlp_f1 = float(mlp_metrics.get("f1", np.nan))
    mlp_auc = float(mlp_metrics.get("auc", np.nan))
    mlp_params = best_params.get("mlp_keras", {})
    bonus_rows = int(len(bonus_mlp_tuning_df))
    bonus_best_row = (
        bonus_mlp_tuning_df.sort_values("val_f1", ascending=False).iloc[0]
        if bonus_rows > 0 and "val_f1" in bonus_mlp_tuning_df.columns
        else None
    )
    bonus_top3 = (
        bonus_mlp_tuning_df.sort_values("val_f1", ascending=False).head(3).copy()
        if bonus_rows > 0 and "val_f1" in bonus_mlp_tuning_df.columns
        else pd.DataFrame()
    )
    if not bonus_top3.empty:
        bonus_top3 = bonus_top3[
            ["hidden_layers", "learning_rate", "dropout_rate", "val_f1", "epochs_trained", "best_val_loss"]
        ].rename(
            columns={
                "hidden_layers": "Hidden Layers",
                "learning_rate": "Learning Rate",
                "dropout_rate": "Dropout",
                "val_f1": "Validation F1",
                "epochs_trained": "Epochs Trained",
                "best_val_loss": "Best Val Loss",
            }
        )
    best_row_by_f1 = comparison_df.sort_values("f1", ascending=False).iloc[0]
    best_model_name = str(best_row_by_f1["model"])
    best_model_f1 = float(best_row_by_f1["f1"])

    st.subheader("2.1 Data Preparation")
    st.markdown(
        f"**X and y definition:** `y = Dropout_flag` (1 = Dropout, 0 = Non-dropout) and "
        f"`X = final train-only rechecked feature set` (`n={len(feature_names)}` features)."
    )
    st.markdown(
        f"**Train/test split:** stratified **70/30** split with `random_state=42` "
        f"(current dataset: total `{total_n}`, train `{train_n}`, test `{test_n}`), performed before model tuning."
    )
    st.markdown(
        "**Preprocessing:** All predictors in this dataset are already integer-coded, so no one-hot encoding was needed. "
        "Median imputation was applied to handle any missing values without dropping rows. "
        "`StandardScaler` was applied for Logistic Regression and MLP since those models are sensitive to feature scale; "
        "tree-based models only use imputation. "
        "All preprocessing was fit on training data only; test data is never seen until final evaluation."
    )
    st.markdown(
        f"**Final selected features ({len(feature_names)} total):** `{', '.join(feature_names)}`"
    )
    st.markdown("**Why 10 features? (Train-only recheck)**")
    st.markdown(
        "Starting from an initial 18-feature list, I rechecked features on the training split only: "
        "first a univariate logistic screen (keeping features with `p < 0.05`), then correlation filtering (removing features with `|r| > 0.75` against already-selected ones). "
        "That gave a rechecked 18-feature set. "
        "I then built a 10-feature candidate by ranking absolute logistic coefficients within that set "
        "and compared the two using 5-fold stratified CV with F1 as the criterion."
    )
    if not np.isnan(cv_f1_18) and not np.isnan(cv_f1_10):
        cv_diff = cv_f1_10 - cv_f1_18
        st.markdown(
            f"CV result: **F1(18) = {cv_f1_18:.3f}** vs **F1(10) = {cv_f1_10:.3f}** "
            f"(difference: **{cv_diff:+.3f}** toward the selected set)."
        )
        st.markdown(
            f"Feature-set decision: the gap is small (**{cv_diff:+.3f}**), but by the predefined rule "
            "we select the set with higher CV F1, so the **10-feature set is adopted as final**."
        )
    if feature_choice_logic:
        st.caption(f"Selection rule: {feature_choice_logic}")
    with st.expander("Feature-selection details (18-feature and 10-feature candidate lists)"):
        st.markdown(
            f"**Rechecked 18-feature set** (`n={len(selected_18_rechecked)}`): "
            + (f"`{', '.join(selected_18_rechecked)}`" if selected_18_rechecked else "`N/A`")
        )
        st.markdown(
            f"**10-feature candidate** (`n={len(selected_10_candidate)}`): "
            + (f"`{', '.join(selected_10_candidate)}`" if selected_10_candidate else "`N/A`")
        )
        st.markdown(
            f"**Final deployed set** (`n={len(feature_names)}`): "
            + (f"`{', '.join(feature_names)}`" if feature_names else "`N/A`")
        )
    st.markdown(
        "Tuning/evaluation protocol: Decision Tree, Random Forest, and LightGBM are tuned with **5-fold Stratified CV** "
        "on the training split only (`random_state=42`); MLP hyperparameters are tuned with a **train-only hold-out validation split**; "
        "final model metrics are reported once on the untouched **30% test set**."
    )

    st.subheader("Imbalanced-Data Handling")
    st.markdown(
        f"The binary target is imbalanced: **Dropout = {eda_highlights['dropout_rate']:.1%}** vs "
        f"**Non-dropout = {(1-eda_highlights['dropout_rate']):.1%}**. "
        "To handle this, I used a stratified split to preserve the same class ratio in both train and test sets, "
        "and applied class weights so each model is penalized more heavily for missing actual dropout cases "
        "(`class_weight='balanced'` for logistic/tree/forest/LightGBM; computed class weights for MLP). "
        "Model selection relied on F1 and AUC rather than accuracy alone, "
        "and I reviewed precision/recall tradeoffs for each model to understand the practical implications. "
        "Random oversampling was not used here because weighted objectives already improved minority-class sensitivity "
        "without the added complexity."
    )

    st.subheader("2.2 Logistic Regression Baseline")
    st.markdown(
        "Classification baseline model is Logistic Regression. "
        "The test report includes Accuracy, Precision, Recall, F1, and AUC-ROC."
    )
    st.dataframe(one_row_metrics("logistic", "Logistic Regression (Baseline)"), hide_index=True, use_container_width=True)
    st.image(str(FIGURES / "part2_roc_logistic.png"), use_container_width=True)
    st.markdown(
        f"Result interpretation: Logistic baseline reaches **Accuracy {baseline_accuracy:.3f}**, **Precision {baseline_precision:.3f}**, "
        f"**Recall {baseline_recall:.3f}**, **F1 {baseline_f1:.3f}**, and **AUC {baseline_auc:.3f}** on the held-out test set."
    )
    st.markdown(
        "This is a strong baseline for this dataset: recall is relatively high, so the model catches many at-risk students, "
        "while AUC above 0.90 indicates good ranking ability across thresholds. "
        "Precision is lower than recall, which means the model is intentionally more sensitive (more alerts, including some false positives)."
    )
    st.markdown(
        "Baseline role: this logistic model is the reference point for comparing later models. "
        "Cross-model improvement is evaluated in Section 2.7 after all candidate models are presented."
    )

    st.subheader("2.3 Decision Tree (GridSearchCV, 5-fold)")
    st.markdown(
        "Tuning grid: `max_depth = [3, 5, 7, 10]`, `min_samples_leaf = [5, 10, 20, 50]`, `scoring = F1`."
    )
    st.markdown(f"Best params: `{best_params.get('decision_tree', {})}`")
    st.dataframe(one_row_metrics("decision_tree", "Decision Tree"), hide_index=True, use_container_width=True)
    st.image(str(FIGURES / "part2_best_decision_tree.png"), use_container_width=True)
    st.image(str(FIGURES / "part2_roc_decision_tree.png"), use_container_width=True)
    st.markdown(
        f"Result interpretation: the best CV setting is `max_depth={dt_params.get('model__max_depth', 'N/A')}` "
        f"and `min_samples_leaf={dt_params.get('model__min_samples_leaf', 'N/A')}`. "
        "This favors a relatively simple tree with larger leaves, which helps control overfitting."
    )
    st.markdown(
        f"On the test set, Decision Tree reaches **Accuracy {dt_accuracy:.3f}**, **Precision {dt_precision:.3f}**, "
        f"**Recall {dt_recall:.3f}**, **F1 {dt_f1:.3f}**, and **AUC {dt_auc:.3f}**."
    )
    if not np.isnan(baseline_f1):
        st.markdown(
            f"Baseline comparison: vs Logistic, Decision Tree improves F1 by **{dt_f1 - baseline_f1:+.3f}** "
            f"and changes AUC by **{dt_auc - baseline_auc:+.3f}**. "
            "In practical terms, it slightly improves minority-class capture while remaining interpretable through explicit split rules."
        )
    if dt_level_notes:
        st.markdown("**Tree-structure interpretation (first levels):**")
        st.markdown("\n".join([f"{i}. {line}" for i, line in enumerate(dt_level_notes, start=1)]))
        st.markdown(
            "Reading guide: depth 0 is the first split (root), depth 1 is the second layer, and depth 2 is the next layer. "
            "Use these top levels as the model's main screening logic."
        )
    if dt_path_summary:
        high = dt_path_summary["high_risk"]
        low = dt_path_summary["low_risk"]
        root_share = float(dt_path_summary["root_dropout_share"])
        high_share = float(high["dropout_share"])
        low_share = float(low["dropout_share"])
        st.markdown("**What these splits mean in practice:**")
        st.markdown(
            f"The overall dropout share at the root (before any splits) is **{root_share:.1%}**. "
            f"Following the high-risk path (`{high['path']}`), the leaf-level dropout share reaches **{high_share:.1%}**. "
            f"The low-risk path (`{low['path']}`) drops to just **{low_share:.1%}**."
        )
        st.markdown(
            f"Interpretation: the gap between high-risk and low-risk leaves is **{high_share - low_share:+.1%}**. "
            "This means the CART model is not just fitting noise; it is separating meaningful student profiles into clearly different risk groups. "
            "Operationally, the high-risk path can be used as an early-alert rule template, while the low-risk path indicates conditions associated with stable retention."
        )

    st.subheader("2.4 Random Forest (GridSearchCV, 5-fold)")
    st.markdown(
        "Tuning grid: `n_estimators = [50, 100, 200]`, `max_depth = [3, 5, 8]`, `scoring = F1`."
    )
    st.markdown(f"Best params: `{best_params.get('random_forest', {})}`")
    st.dataframe(one_row_metrics("random_forest", "Random Forest"), hide_index=True, use_container_width=True)
    st.image(str(FIGURES / "part2_roc_random_forest.png"), use_container_width=True)
    st.markdown(
        "How best hyperparameters were found and reported: "
        "we run `GridSearchCV` on the training split only with `5-fold Stratified CV`, "
        "evaluate all `3 x 3 = 9` parameter combinations using mean CV F1, "
        "and then refit the best setting on the full training data."
    )
    st.markdown(
        f"Reported tuning outcome: best hyperparameters are `n_estimators={rf_params.get('model__n_estimators', 'N/A')}` "
        f"and `max_depth={rf_params.get('model__max_depth', 'N/A')}`. "
        f"Final held-out test metrics are Accuracy **{rf_accuracy:.3f}**, Precision **{rf_precision:.3f}**, "
        f"Recall **{rf_recall:.3f}**, F1 **{rf_f1:.3f}**, and AUC **{rf_auc:.3f}**."
    )
    st.markdown(
        f"Result interpretation: the selected setting is `n_estimators={rf_params.get('model__n_estimators', 'N/A')}` "
        f"and `max_depth={rf_params.get('model__max_depth', 'N/A')}`. "
        "This balance usually gives enough model capacity to capture nonlinear risk patterns without becoming too deep."
    )
    st.markdown(
        f"On the test set, Random Forest reaches **Accuracy {rf_accuracy:.3f}**, **Precision {rf_precision:.3f}**, "
        f"**Recall {rf_recall:.3f}**, **F1 {rf_f1:.3f}**, and **AUC {rf_auc:.3f}**."
    )
    if not np.isnan(baseline_f1):
        st.markdown(
            f"Compared with the Logistic baseline, Random Forest changes F1 by **{rf_f1 - baseline_f1:+.3f}** "
            f"and AUC by **{rf_auc - baseline_auc:+.3f}**. "
            "The pattern here is straightforward: precision improves while recall stays competitive, "
            "so the model reduces unnecessary alerts while still identifying most at-risk students."
        )

    st.subheader("2.5 Boosted Tree (LightGBM, GridSearchCV, 5-fold)")
    st.markdown(
        "Boosted-tree implementation uses LightGBM. "
        "Tuning covers at least 3 hyperparameters: "
        "`n_estimators = [50, 100, 200]`, `max_depth = [3, 4, 5, 6]`, `learning_rate = [0.01, 0.05, 0.1]` with `scoring = F1`."
    )
    st.markdown(f"Best params: `{best_params.get('lightgbm', {})}`")
    st.dataframe(one_row_metrics("lightgbm", "Boosted Tree (LightGBM)"), hide_index=True, use_container_width=True)
    st.image(str(FIGURES / "part2_roc_lightgbm.png"), use_container_width=True)
    st.markdown(
        "How best hyperparameters were found and reported: "
        "we run `GridSearchCV` on training data only with `5-fold Stratified CV`, "
        "evaluate all `3 x 4 x 3 = 36` combinations by mean CV F1, "
        "and refit the best LightGBM setting on the full training split."
    )
    st.markdown(
        f"Reported tuning outcome: best hyperparameters are `n_estimators={lgbm_params.get('model__n_estimators', 'N/A')}`, "
        f"`max_depth={lgbm_params.get('model__max_depth', 'N/A')}`, and "
        f"`learning_rate={lgbm_params.get('model__learning_rate', 'N/A')}`. "
        f"Final held-out test metrics are Accuracy **{lgbm_accuracy:.3f}**, Precision **{lgbm_precision:.3f}**, "
        f"Recall **{lgbm_recall:.3f}**, F1 **{lgbm_f1:.3f}**, and AUC **{lgbm_auc:.3f}**."
    )
    st.markdown(
        f"Result interpretation: the selected setting is `n_estimators={lgbm_params.get('model__n_estimators', 'N/A')}`, "
        f"`max_depth={lgbm_params.get('model__max_depth', 'N/A')}`, and "
        f"`learning_rate={lgbm_params.get('model__learning_rate', 'N/A')}`. "
        "This setup favors controlled boosting steps, so each tree adds signal without overreacting to noise."
    )
    st.markdown(
        f"On the test set, LightGBM reaches **Accuracy {lgbm_accuracy:.3f}**, **Precision {lgbm_precision:.3f}**, "
        f"**Recall {lgbm_recall:.3f}**, **F1 {lgbm_f1:.3f}**, and **AUC {lgbm_auc:.3f}**."
    )
    if not np.isnan(baseline_f1):
        st.markdown(
            f"Compared with the Logistic baseline, LightGBM changes F1 by **{lgbm_f1 - baseline_f1:+.3f}** "
            f"and AUC by **{lgbm_auc - baseline_auc:+.3f}**. "
            "The practical takeaway is simple: this model gives the strongest probability ranking in this project "
            "(highest AUC), which helps prioritize intervention lists when resources are limited."
        )

    st.subheader("2.6 Neural Network (Keras MLP)")
    st.markdown(
        f"Framework: **Keras (TensorFlow backend)**. "
        f"Input layer size matches the final feature dimension (`input_dim = {len(feature_names)}`). "
        "We build the network with two dense ReLU hidden layers and a sigmoid output, then tune hidden width/dropout/learning-rate and keep the best validation-F1 configuration."
    )
    st.markdown(
        "Training setup: `binary_crossentropy` loss, `Adam` optimizer, class weights for imbalance handling, "
        "early stopping on validation loss, and training-history tracking (loss + accuracy curves)."
    )
    st.markdown(f"Best tuned MLP config: `{mlp_params}`")
    st.markdown("**Test-set metrics report (held-out 30% test set):**")
    st.markdown(
        "The table below reports the same five metrics used for all classification models: "
        "**Accuracy, Precision, Recall, F1, and AUC-ROC**."
    )
    st.dataframe(one_row_metrics("mlp_keras", "MLP (Keras)"), hide_index=True, use_container_width=True)
    mlp_col1, mlp_col2 = st.columns(2)
    with mlp_col1:
        st.image(str(FIGURES / "part2_mlp_training_history.png"), use_container_width=True)
    with mlp_col2:
        st.image(str(FIGURES / "part2_roc_mlp_keras.png"), use_container_width=True)
    st.markdown(
        f"Result interpretation: on the test set, the MLP reaches **Accuracy {mlp_accuracy:.3f}**, "
        f"**Precision {mlp_precision:.3f}**, **Recall {mlp_recall:.3f}**, **F1 {mlp_f1:.3f}**, and **AUC {mlp_auc:.3f}**."
    )
    if not np.isnan(baseline_f1):
        st.markdown(
            f"Compared with Logistic baseline, MLP changes F1 by **{mlp_f1 - baseline_f1:+.3f}** "
            f"and AUC by **{mlp_auc - baseline_auc:+.3f}**. "
            "In this project, the MLP gives strong ranking quality but slightly lower F1 than the best tree models, "
            "so it is useful as a nonlinear benchmark rather than the final deployment winner."
        )

    st.subheader("MLP Hyperparameter Tuning")
    st.image(str(FIGURES / "bonus_mlp_tuning_heatmap.png"), use_container_width=True)
    st.markdown(
        "Heatmap reading tip: each cell is one hyperparameter combination, and darker cells indicate higher validation F1."
    )
    st.markdown(
        "Tuning approach: we performed a structured grid search over hidden-layer width patterns, "
        "learning rates, and dropout rates using a train-only validation split and validation F1 as the selection criterion."
    )
    st.markdown(
        "Search space: hidden layers `[(64,64), (128,128), (128,64)]` x learning rate `[0.001, 0.0005]` x "
        "dropout `[0.0, 0.2]` = **12 candidate configurations**."
    )
    st.markdown(
        "**How to read the table:** Hidden Layers `(a, b)` means two hidden layers where the first has `a` neurons and the second has `b`. "
        "Learning Rate controls the Adam optimizer step size (`0.001` is faster, `0.0005` is slower but often steadier). "
        "Dropout is the fraction of hidden units randomly turned off during each training step (0.2 means 20%). "
        "Validation F1 is the tuning criterion (higher is better). "
        "Epochs Trained shows how many training rounds ran before early stopping triggered. "
        "Best Val Loss is the lowest validation loss observed during training; lower indicates a better fit."
    )
    if bonus_best_row is not None:
        st.markdown(
            f"Best validation result: hidden layers **{bonus_best_row['hidden_layers']}**, learning rate **{float(bonus_best_row['learning_rate']):.4f}**, "
            f"dropout **{float(bonus_best_row['dropout_rate']):.1f}**, validation F1 **{float(bonus_best_row['val_f1']):.3f}**."
        )
    st.markdown(
        "Tuning insight: lower learning rate (`0.0005`) with moderate dropout (`0.2`) was more stable in this dataset, "
        "which suggests regularization helps generalization when predicting minority-risk students."
    )
    if not bonus_top3.empty:
        st.dataframe(bonus_top3, hide_index=True, use_container_width=True)

    st.subheader("2.7 Model Comparison Summary")
    model_name_map = {
        "logistic": "Logistic Regression (Baseline)",
        "decision_tree": "Decision Tree (CART)",
        "random_forest": "Random Forest",
        "lightgbm": "LightGBM (Boosted Tree)",
        "mlp_keras": "MLP (Keras)",
    }
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    summary_df = comparison_df.copy()
    summary_df["Model"] = summary_df["model"].map(model_name_map).fillna(summary_df["model"])
    summary_df = (
        summary_df[["Model", "accuracy", "precision", "recall", "f1", "auc"]]
        .rename(
            columns={
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "auc": "AUC-ROC",
            }
        )
        .sort_values("F1", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    st.markdown("**Interactive metric explorer:**")
    selected_metric = st.selectbox(
        "Choose metric for interactive comparison",
        metric_cols,
        index=3,
        key="metric_explorer_tab3",
    )
    metric_chart_df = summary_df[["Model", selected_metric]].sort_values(selected_metric, ascending=False).copy()
    metric_chart = (
        alt.Chart(metric_chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Model:N", sort="-y", axis=alt.Axis(labelAngle=-20)),
            y=alt.Y(f"{selected_metric}:Q", title=selected_metric, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(f"{selected_metric}:Q", legend=None, scale=alt.Scale(scheme="tealblues")),
            tooltip=[alt.Tooltip("Model:N"), alt.Tooltip(f"{selected_metric}:Q", format=".3f")],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(metric_chart, use_container_width=True)

    # F1 chart with zoomed y-range so close scores are still visually distinguishable.
    fig_f1, ax_f1 = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=summary_df, x="Model", y="F1", palette="Set2", ax=ax_f1)
    f1_min = float(summary_df["F1"].min())
    f1_max = float(summary_df["F1"].max())
    ax_f1.set_ylim(max(0.0, f1_min - 0.015), min(1.0, f1_max + 0.015))
    ax_f1.set_title("F1 Comparison (Zoomed Scale for Small Differences)")
    ax_f1.set_xlabel("")
    ax_f1.set_ylabel("F1")
    ax_f1.tick_params(axis="x", rotation=18)
    for idx, val in enumerate(summary_df["F1"].tolist()):
        ax_f1.text(idx, float(val) + 0.0012, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax_f1.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig_f1, clear_figure=True)
    plt.close(fig_f1)
    st.image(str(FIGURES / "part2_f1_bar_comparison.png"), use_container_width=True)
    st.caption("Saved Part 2 F1 comparison artifact from the training pipeline.")

    # Metric heatmap for compact side-by-side comparison.
    heat_df = summary_df.set_index("Model")[metric_cols]
    fig_heat, ax_heat = plt.subplots(figsize=(10.5, 4.6))
    sns.heatmap(
        heat_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=float(heat_df.values.min()),
        vmax=float(heat_df.values.max()),
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"shrink": 0.85},
        ax=ax_heat,
    )
    ax_heat.set_title("Model-Metric Heatmap")
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig_heat, clear_figure=True)
    plt.close(fig_heat)

    # Baseline-delta view (how each model moves vs Logistic baseline).
    baseline_label = "Logistic Regression (Baseline)"
    if baseline_label in heat_df.index:
        baseline_vec = heat_df.loc[baseline_label]
        delta_df = heat_df.subtract(baseline_vec, axis=1).drop(index=[baseline_label], errors="ignore")
        fig_delta, ax_delta = plt.subplots(figsize=(10.5, 3.9))
        vmax = float(np.abs(delta_df.values).max()) if len(delta_df) else 0.01
        vmax = max(vmax, 0.005)
        sns.heatmap(
            delta_df,
            annot=True,
            fmt="+.3f",
            cmap="RdBu_r",
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.2,
            linecolor="white",
            cbar_kws={"shrink": 0.8},
            ax=ax_delta,
        )
        ax_delta.set_title("Improvement vs Logistic Baseline (Positive Is Better)")
        ax_delta.set_xlabel("")
        ax_delta.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig_delta, clear_figure=True)
        plt.close(fig_delta)

    # Rank table by metric.
    rank_df = heat_df.rank(ascending=False, method="min").astype(int)
    rank_df = rank_df.rename(columns={c: f"{c} Rank" for c in rank_df.columns})
    st.dataframe(rank_df.reset_index(), hide_index=True, use_container_width=True)

    # Detailed metric-by-metric interpretation.
    best_acc_row = summary_df.loc[summary_df["Accuracy"].idxmax()]
    best_prec_row = summary_df.loc[summary_df["Precision"].idxmax()]
    best_rec_row = summary_df.loc[summary_df["Recall"].idxmax()]
    best_f1_row = summary_df.loc[summary_df["F1"].idxmax()]
    best_auc_row = summary_df.loc[summary_df["AUC-ROC"].idxmax()]

    f1_winner = str(best_f1_row["Model"])
    auc_winner = str(best_auc_row["Model"])

    st.markdown(
        f"Looking at each metric in turn: the highest accuracy goes to **{best_acc_row['Model']} ({best_acc_row['Accuracy']:.3f})**, "
        "though accuracy alone can be misleading here because the classes are imbalanced; a model that just predicts 'not dropout' most of the time can still look reasonably accurate. "
        f"For precision, **{best_prec_row['Model']} ({best_prec_row['Precision']:.3f})** is strongest, which matters when intervention resources are limited and false alarms are costly. "
        f"For recall (catching actual dropout cases), **{best_rec_row['Model']} ({best_rec_row['Recall']:.3f})** leads, which is valuable when the cost of missing a truly at-risk student is high. "
        f"F1 (the primary criterion here, since it balances precision and recall) is best for **{best_f1_row['Model']} ({best_f1_row['F1']:.3f})**. "
        f"And **{best_auc_row['Model']} ({best_auc_row['AUC-ROC']:.3f})** leads on AUC, which measures how well the model rank-orders students by risk across all possible thresholds."
    )
    if baseline_label in heat_df.index:
        base = heat_df.loc[baseline_label]
        st.markdown(
            f"Compared to the logistic baseline, the best-F1 model (**{best_f1_row['Model']}**) moves "
            f"Accuracy by **{best_f1_row['Accuracy'] - base['Accuracy']:+.3f}**, Precision by **{best_f1_row['Precision'] - base['Precision']:+.3f}**, "
            f"Recall by **{best_f1_row['Recall'] - base['Recall']:+.3f}**, F1 by **{best_f1_row['F1'] - base['F1']:+.3f}**, "
            f"and AUC by **{best_f1_row['AUC-ROC'] - base['AUC-ROC']:+.3f}**. "
            "The default recommendation is to use the top-F1 model as the operational baseline, "
            "then revisit the threshold monthly based on how many false alerts the advising team can actually handle."
        )
    surprise_line = (
        f"One interesting thing about this run: **{auc_winner}** had the top AUC ({best_auc_row['AUC-ROC']:.3f}) "
        f"while **{f1_winner}** still led F1 ({best_f1_row['F1']:.3f}), meaning ranking quality and fixed-threshold balance diverged slightly."
        if f1_winner != auc_winner
        else f"Both F1 and AUC ended up pointing to the same model, **{f1_winner}**, which makes the choice fairly clear."
    )
    st.markdown(
        f"Overall, **{f1_winner}** is the best-performing model in this run by the primary F1 criterion. "
        f"{surprise_line} "
        "The broader tradeoff across models is worth understanding: logistic regression and a single decision tree are fast, transparent, and easy to explain to non-technical stakeholders, "
        "but they give up some predictive power on nonlinear patterns. "
        "Random Forest and LightGBM pick up those nonlinear interactions at the cost of interpretability and longer tuning time. "
        "The neural network (MLP) was actually the one that surprised me most; I expected it to do better, "
        "but this dataset has a lot of threshold-style interactions (e.g., specific grade cutpoints combined with financial stress flags) "
        "that tree ensembles handle very naturally, while the MLP needs careful regularization and architecture choices to catch them. "
        "The MLP is also the hardest of the five models to explain to someone who isn't technical."
    )
    st.subheader("Final Model Decision")
    st.markdown(
        f"My final model choice for this project is **{f1_winner}**, "
        f"which achieved the best test-set F1 (**{best_f1_row['F1']:.3f}**) with solid precision (**{best_f1_row['Precision']:.3f}**), "
        f"recall (**{best_f1_row['Recall']:.3f}**), and AUC (**{best_f1_row['AUC-ROC']:.3f}**). "
        "It's the best fit for the goal of catching at-risk students while keeping false alerts at a manageable level."
    )
    if f1_winner != auc_winner:
        st.markdown(
            f"**{auc_winner}** is worth keeping as an alternative when the priority is pure risk-ranking quality across all possible thresholds, given that its AUC of {best_auc_row['AUC-ROC']:.3f} is the highest in the comparison."
        )

    st.subheader("Best Hyperparameters (All Models)")
    st.json(best_params)

    st.subheader("ROC Curves (All Models)")
    roc_items = [
        ("Logistic", "part2_roc_logistic.png"),
        ("Decision Tree (CART)", "part2_roc_decision_tree.png"),
        ("Random Forest", "part2_roc_random_forest.png"),
        ("LightGBM", "part2_roc_lightgbm.png"),
        ("MLP (Keras)", "part2_roc_mlp_keras.png"),
    ]
    row1 = st.columns(3)
    for col, (label, fn) in zip(row1, roc_items[:3]):
        with col:
            st.markdown(f"**{label}**")
            st.image(str(FIGURES / fn), use_container_width=True)
    row2 = st.columns(2)
    for col, (label, fn) in zip(row2, roc_items[3:]):
        with col:
            st.markdown(f"**{label}**")
            st.image(str(FIGURES / fn), use_container_width=True)
    st.markdown("**Interactive ROC explorer (model selector + hover):**")
    roc_model_selected = st.selectbox(
        "Choose model for interactive ROC view",
        summary_df["Model"].tolist(),
        index=0,
        key="roc_explorer_tab3",
    )
    roc_selected_df = roc_points_df[roc_points_df["Model"] == roc_model_selected].copy()
    roc_diagonal_df = pd.DataFrame(
        {
            "False Positive Rate": [0.0, 1.0],
            "True Positive Rate": [0.0, 1.0],
        }
    )
    roc_line = (
        alt.Chart(roc_selected_df)
        .mark_line(strokeWidth=2.5, color="#1f77b4")
        .encode(
            x=alt.X("False Positive Rate:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("True Positive Rate:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("False Positive Rate:Q", format=".3f"),
                alt.Tooltip("True Positive Rate:Q", format=".3f"),
            ],
        )
    )
    roc_diag = (
        alt.Chart(roc_diagonal_df)
        .mark_line(color="#999999", strokeDash=[6, 4])
        .encode(x="False Positive Rate:Q", y="True Positive Rate:Q")
    )
    st.altair_chart((roc_diag + roc_line).properties(height=330).interactive(), use_container_width=True)
    auc_rank = summary_df[["Model", "AUC-ROC"]].sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    if len(auc_rank) >= 2:
        auc_gap = float(auc_rank.loc[0, "AUC-ROC"] - auc_rank.loc[1, "AUC-ROC"])
        st.markdown(
            f"ROC interpretation: the highest AUC is **{auc_rank.loc[0, 'Model']} ({auc_rank.loc[0, 'AUC-ROC']:.3f})**, "
            f"followed by **{auc_rank.loc[1, 'Model']} ({auc_rank.loc[1, 'AUC-ROC']:.3f})** (gap **{auc_gap:.3f}**). "
            "This means both models rank higher-risk students well, but the top model gives slightly cleaner separation across thresholds."
        )
    st.markdown(
        "How to read these curves: lines closer to the top-left indicate stronger discrimination. "
        "Curves closer to the diagonal indicate weaker ranking power. "
        "If your intervention team prefers catching as many at-risk students as possible, use a lower threshold (higher recall, more false positives). "
        "If resources are limited and false alerts are costly, raise the threshold to prioritize precision."
    )


with tab4:
    st.header("Part 3: SHAP Explainability")

    model_label = {
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "lightgbm": "LightGBM",
        "logistic": "Logistic Regression",
        "mlp_keras": "MLP (Keras)",
    }

    st.subheader("3.1 SHAP Analysis (Best-performing Tree-based Model)")
    tree_models = ["decision_tree", "random_forest", "lightgbm"]
    tree_perf = comparison_df[comparison_df["model"].isin(tree_models)].copy()
    tree_perf = tree_perf.sort_values("f1", ascending=False).reset_index(drop=True)
    tree_perf["Model"] = tree_perf["model"].map(model_label).fillna(tree_perf["model"])
    tree_perf_view = tree_perf[["Model", "accuracy", "precision", "recall", "f1", "auc"]].rename(
        columns={
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "auc": "AUC-ROC",
        }
    )
    st.dataframe(tree_perf_view, hide_index=True, use_container_width=True)

    chosen_tree_row = tree_perf.loc[tree_perf["model"] == best_tree_model].iloc[0]
    best_auc_tree_row = tree_perf.sort_values("auc", ascending=False).iloc[0]
    st.markdown(
        f"SHAP is run on **{model_label.get(best_tree_model, best_tree_model)}**, because it is the best tree-based model by F1 on the held-out test set "
        f"(**F1 = {chosen_tree_row['f1']:.3f}**, **AUC = {chosen_tree_row['auc']:.3f}**). "
        "This project prioritizes F1 for dropout-risk detection quality under class imbalance."
    )
    if str(best_auc_tree_row["model"]) != best_tree_model:
        st.markdown(
            f"Note: **{model_label.get(str(best_auc_tree_row['model']), str(best_auc_tree_row['model']))}** has the highest tree-model AUC "
            f"(**{best_auc_tree_row['auc']:.3f}**), but SHAP base model selection follows the F1-first rule defined in the pipeline."
        )
    st.markdown(
        "The SHAP analysis starts by selecting the best tree-based model from the held-out test comparison, "
        "then builds a TreeExplainer on top of it. "
        "SHAP values are computed on the test data to measure how much each feature contributed to each individual prediction. "
        "The output includes a global summary across all students (beeswarm and bar plots) "
        "and one local case study (waterfall plot) to show how the contributions add up for a single high-risk student."
    )

    st.subheader("3.2 Global SHAP Plots")
    st.image(str(FIGURES / "part3_shap_summary_beeswarm.png"), use_container_width=True)
    st.markdown(
        "**How to read the beeswarm plot:** each dot is one student. "
        "Dots on the right push the prediction toward dropout risk; dots on the left push it away. "
        "Color shows feature value level (high vs low), so you can see whether high values increase or decrease risk."
    )
    st.image(str(FIGURES / "part3_shap_bar.png"), use_container_width=True)
    st.markdown(
        "**How to read the bar plot:** features are ranked by mean absolute SHAP value. "
        "Higher bars mean the feature changes predictions more on average across the population."
    )

    shap_rank_df = pd.DataFrame(shap_interpretation)
    if not shap_rank_df.empty:
        shap_rank_df = shap_rank_df.copy()
        shap_rank_df["Rank"] = np.arange(1, len(shap_rank_df) + 1)
        shap_rank_df["mean|SHAP|"] = shap_rank_df["mean_abs_shap"]
        shap_rank_df["Direction"] = shap_rank_df["direction"]
        st.dataframe(
            shap_rank_df[["Rank", "feature", "mean|SHAP|", "Direction"]].rename(columns={"feature": "Feature"}),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("3.3 SHAP Interpretation")
    top_feature_names = ", ".join([f"`{x['feature']}`" for x in shap_interpretation])
    direction_sentence = ". ".join([f"`{x['feature']}`: {x['direction']}" for x in shap_interpretation])
    st.markdown(
        f"**Which features matter most?** The top drivers in this run are {top_feature_names}. "
        "These show up consistently in both the beeswarm and the mean |SHAP| bar ranking."
    )
    st.markdown(
        f"**How do they influence predictions?** {direction_sentence}. "
        "The model is essentially combining academic-performance signals with financial-stability signals to push risk up or down, "
        "and when both are unfavorable at the same time, the combined effect is substantially larger than either alone."
    )
    st.markdown(
        "**What can decision-makers actually do with this?** "
        "The most practical use is generating a risk-ranked list at the start of each term, before withdrawal deadlines hit. "
        "Advisors can work through the high-risk tier first, using the SHAP drivers to understand what kind of support each student actually needs. "
        "A student flagged mainly because of low first-semester grades needs a different response than one flagged primarily because of overdue tuition: "
        "the former might need tutoring and academic coaching, while the latter needs payment-plan counseling or emergency aid referral. "
        "When both signals appear together, that's when coordinated outreach from both advising and financial-aid teams matters most, "
        "because those students are often facing a compounding problem that a single intervention won't fully address."
    )
    st.markdown(
        "Faculty alerts for low first-semester grades can be triggered very early in the term, giving the institution more time to act. "
        "Financial signals like tuition status and debtor flags can be monitored by financial-aid teams as a separate early-warning channel. "
        "The SHAP waterfall plots make each individual prediction auditable: an advisor can show a student exactly what factors went into their risk score, "
        "which makes the conversation more transparent and helps avoid the 'black box' feeling that often makes people skeptical of algorithmic tools. "
        "Over time, these outputs can also feed into cross-functional planning: advisors, faculty, student-success services, "
        "financial aid, and institutional research all working from the same risk framework rather than separate silos. "
        "The model should always be used to direct support, not to exclude or penalize students, "
        "and fairness monitoring across demographic groups should be built into any regular review cycle."
    )

    extra_insights: List[str] = []
    shap_feature_set = {x["feature"] for x in shap_interpretation}
    if "Curricular units 2nd sem (approved)" in shap_feature_set:
        extra_insights.append(
            "Academic progression is the dominant risk separator: when approved units in semester 2 drop, SHAP contributions shift toward dropout consistently across students."
        )
    if "Tuition fees up to date" in shap_feature_set:
        extra_insights.append(
            "Tuition status behaves almost like a switch: being up to date pushes risk down, while not being up to date pushes it up, often quite sharply."
        )
    if "Curricular units 1st sem (grade)" in shap_feature_set:
        extra_insights.append(
            "First-semester grade still adds meaningful signal even after second-semester variables are in the model, because early performance leaves a lasting imprint on the prediction."
        )
    if "Age at enrollment" in shap_feature_set:
        extra_insights.append(
            "Age at enrollment has a directional effect on average (older tends to mean higher risk), but SHAP shows considerable variation across individuals; it's a signal, not a verdict."
        )
    if "Scholarship holder" in shap_feature_set:
        extra_insights.append(
            "Having a scholarship consistently pulls risk downward in the model, which aligns with the idea that financial security acts as a retention buffer."
        )
    if extra_insights:
        st.markdown("**Additional patterns from SHAP:** " + " ".join(extra_insights))

    st.subheader("3.4 Reference Waterfall Example from Test Set")
    st.image(str(FIGURES / "part3_shap_waterfall_example.png"), use_container_width=True)
    st.markdown(
        "This reference case shows one high-risk student from the test set. "
        "It demonstrates how multiple moderate risk factors can combine into a high final prediction, "
        "not only one single extreme variable."
    )
    if reference_waterfall:
        st.markdown(
            f"Reference-case details: this example is test-set position **{reference_waterfall['reference_test_position']}** "
            f"(original dataframe index **{reference_waterfall['reference_index']}**), "
            f"with predicted dropout probability **{reference_waterfall['reference_probability']:.3f}** "
            f"from **{model_label.get(best_tree_model, best_tree_model)}**."
        )
        risk_df = pd.DataFrame(reference_waterfall["top_risk"])
        protect_df = pd.DataFrame(reference_waterfall["top_protective"])
        if not risk_df.empty:
            st.markdown("**Top risk-increasing contributors in this case (largest positive SHAP):**")
            st.dataframe(risk_df, hide_index=True, use_container_width=True)
        if not protect_df.empty:
            st.markdown("**Top risk-reducing contributors in this case (largest negative SHAP):**")
            st.dataframe(protect_df, hide_index=True, use_container_width=True)
        st.markdown(
            "Decision-maker interpretation for this case: if risk-increasing drivers are mainly academic, "
            "prioritize tutoring/course-load intervention; if financial drivers dominate, prioritize payment counseling/support; "
            "if both appear together, assign integrated support quickly because combined drivers usually indicate higher persistence risk."
        )

    st.subheader("3.5 Interactive Prediction + Custom SHAP Waterfall")
    model_options = ["logistic", "decision_tree", "random_forest", "lightgbm", "mlp_keras"]
    default_idx = model_options.index(best_tree_model) if best_tree_model in model_options else 2
    selected_model = st.selectbox("Select model for prediction", model_options, index=default_idx)
    prediction_threshold = st.slider(
        "Classification threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
        help="If predicted dropout probability is above this threshold, class is set to Dropout (1).",
    )
    auto_update = st.toggle("Auto-update prediction as inputs change", value=False)
    st.caption(
        f"Manual inputs are enabled for **{len(manual_input_features)} key features**; "
        "all other features are fixed at dataset-average values."
    )

    input_values: Dict[str, float] = {}
    for f in feature_names:
        rng = feature_ranges[f]
        f_min, f_max, f_med = rng["min"], rng["max"], rng["median"]

        if f in manual_input_features:
            # Binary-like fields as dropdowns for cleaner UX.
            if f_min >= 0 and f_max <= 1:
                input_values[f] = float(st.selectbox(f, options=[0, 1], index=int(round(f_med))))
            else:
                if float(f_min).is_integer() and float(f_max).is_integer() and (f_max - f_min) <= 200:
                    input_values[f] = float(
                        st.slider(
                            f,
                            min_value=int(f_min),
                            max_value=int(f_max),
                            value=int(round(f_med)),
                            step=1,
                        )
                    )
                else:
                    input_values[f] = float(
                        st.slider(
                            f,
                            min_value=float(f_min),
                            max_value=float(f_max),
                            value=float(f_med),
                        )
                    )
        else:
            input_values[f] = float(feature_means[f])

    row_df = pd.DataFrame([input_values])

    if "prediction_cache" not in st.session_state:
        st.session_state["prediction_cache"] = None

    run_clicked = st.button("Run Prediction", type="primary")
    if auto_update or run_clicked:
        prob, pred = predict_with_model(
            selected_model,
            row_df,
            feature_names,
            models,
            threshold=float(prediction_threshold),
        )
        if selected_model in {"decision_tree", "random_forest", "lightgbm"}:
            shap_model = selected_model
        else:
            shap_model = best_tree_model
        st.session_state["prediction_cache"] = {
            "prob": float(prob),
            "pred": int(pred),
            "threshold": float(prediction_threshold),
            "selected_model": str(selected_model),
            "shap_model": str(shap_model),
            "input_values": {k: float(v) for k, v in input_values.items()},
        }

    cached = st.session_state.get("prediction_cache")
    if cached is not None:
        prob = float(cached["prob"])
        pred = int(cached["pred"])
        threshold_used = float(cached["threshold"])
        selected_model_used = str(cached["selected_model"])
        shap_model = str(cached["shap_model"])
        row_df_cached = pd.DataFrame([cached["input_values"]])[feature_names]

        pred_label = "Dropout (1)" if pred == 1 else "Non-dropout (0)"
        st.markdown(
            f"Predicted outcome: **{pred_label}** at threshold **{threshold_used:.2f}**."
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Dropout Probability", f"{prob:.1%}")
        with m2:
            st.metric("Non-dropout Probability", f"{1 - prob:.1%}")
        with m3:
            st.metric("Predicted Class", pred_label)

        prob_plot_df = pd.DataFrame(
            {"Probability": [1 - prob, prob]},
            index=["Non-dropout (0)", "Dropout (1)"],
        )
        st.bar_chart(prob_plot_df, use_container_width=True)
        st.caption(
            f"Probability chart for the latest prediction. Current decision threshold is **{threshold_used:.2f}**."
        )

        if selected_model_used in {"decision_tree", "random_forest", "lightgbm"}:
            st.info(
                f"Waterfall explanation uses the selected tree model: **{model_label.get(selected_model_used, selected_model_used)}**."
            )
        else:
            st.info(
                f"Selected model **{model_label.get(selected_model_used, selected_model_used)}** is non-tree. "
                f"For local SHAP explainability, waterfall is generated with the best tree-based model "
                f"**{model_label.get(best_tree_model, best_tree_model)}**."
            )

        fig = make_custom_waterfall(shap_model, row_df_cached, feature_names, models)
        st.pyplot(fig, clear_figure=True)
        st.markdown(
            "Waterfall reading tip: start at the baseline risk, then read bars from top to bottom. "
            "Bars to the right increase dropout risk; bars to the left reduce risk. "
            "The largest bars are the most actionable drivers for this student."
        )
        top_pos, top_neg = compute_custom_shap_contributions(shap_model, row_df_cached, feature_names, models, top_n=5)
        contrib_df = pd.concat([top_pos, top_neg], ignore_index=True).drop_duplicates(subset=["Feature"]).copy()
        if not contrib_df.empty:
            contrib_df["Direction"] = np.where(contrib_df["SHAP Value"] >= 0, "Risk-increasing", "Risk-reducing")
            contrib_chart = (
                alt.Chart(contrib_df)
                .mark_bar()
                .encode(
                    x=alt.X("SHAP Value:Q", title="SHAP Value"),
                    y=alt.Y("Feature:N", sort=alt.SortField(field="SHAP Value", order="descending"), title=""),
                    color=alt.Color("Direction:N", scale=alt.Scale(range=["#d97373", "#4c9f70"])),
                    tooltip=[
                        alt.Tooltip("Feature:N"),
                        alt.Tooltip("Input Value:Q", format=".3f"),
                        alt.Tooltip("SHAP Value:Q", format=".4f"),
                        alt.Tooltip("Direction:N"),
                    ],
                )
                .properties(height=280)
                .interactive()
            )
            st.markdown("**Interactive SHAP contribution chart for this custom input**")
            st.altair_chart(contrib_chart, use_container_width=True)

        c_pos, c_neg = st.columns(2)
        with c_pos:
            st.markdown("**Top risk-increasing drivers for this custom input**")
            st.dataframe(top_pos, hide_index=True, use_container_width=True)
        with c_neg:
            st.markdown("**Top risk-reducing drivers for this custom input**")
            st.dataframe(top_neg, hide_index=True, use_container_width=True)
    else:
        st.info(
            "No prediction has been run yet. Click **Run Prediction** (or turn on auto-update) to generate "
            "the class/probability output and custom SHAP waterfall."
        )
