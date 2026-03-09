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

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        cbar_kws={"shrink": 0.8},
        xticklabels=True,
        yticklabels=True,
        ax=ax,
    )
    ax.set_title("Full Correlation Heatmap (All Numeric Columns)")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    plt.tight_layout()
    return fig


def make_focused_correlation_heatmap_figure(df: pd.DataFrame, corr_cols: List[str]) -> plt.Figure:
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 9))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=False,
        cbar_kws={"shrink": 0.85},
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
    plt.subplots_adjust(bottom=0.22, left=0.26)
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
) -> Tuple[float, int]:
    if model_name == "mlp_keras":
        bundle = models["mlp_bundle"]
        X = bundle["preprocess"].transform(row_df[feature_names])
        prob = float(models["mlp_keras"].predict(X, verbose=0).ravel()[0])
    else:
        prob = float(models[model_name].predict_proba(row_df[feature_names])[:, 1][0])
    pred = int(prob >= 0.5)
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


@st.cache_data
def compute_shap_interpretation(
    _df: pd.DataFrame,
    _feature_names: List[str],
    _best_tree_model: str,
) -> List[Dict[str, Any]]:
    model_map = load_models()
    tree_pipe = model_map[_best_tree_model]
    X = _df[_feature_names].copy()
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


meta = load_metadata()
metrics = load_metrics()
best_params = load_best_params()
comparison_df = load_comparison_df()
df = load_data()
models = load_models()
missing_total = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

feature_names = meta["feature_selection"]["final_features"]
feature_selection_meta = meta.get("feature_selection", {})
selected_18_rechecked = feature_selection_meta.get("selected_18_rechecked", [])
selected_10_candidate = feature_selection_meta.get("selected_10", [])
cv_f1_18 = float(feature_selection_meta.get("cv_f1_18", np.nan))
cv_f1_10 = float(feature_selection_meta.get("cv_f1_10", np.nan))
feature_choice_logic = meta.get("final_model_choice_logic", "")
corr_columns = get_full_corr_columns(df)
corr_full = df[corr_columns].corr()
focus_corr_columns = get_top_target_corr_columns(corr_full, target_col="Dropout_flag", top_k=12)
manual_input_features = meta.get("manual_input_features", [])
feature_ranges = meta["feature_ranges"]
feature_means = meta["feature_means"]
best_tree_model = meta["best_tree_model_for_shap"]
captions = meta["captions"]
top_corr_a, top_corr_b, top_corr_val = strongest_corr_pair(df, corr_columns)
corr_first_second_grade = get_corr_value(corr_full, "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)")
corr_first_second_approved = get_corr_value(corr_full, "Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)")
corr_parent_occ = get_corr_value(corr_full, "Mother's occupation", "Father's occupation")
corr_target_second_grade = get_corr_value(corr_full, "Curricular units 2nd sem (grade)", "Dropout_flag")
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
shap_interpretation = compute_shap_interpretation(df, feature_names, best_tree_model)
eda_highlights = compute_eda_highlights(df)
tradeoff_text = build_model_tradeoff_text(comparison_df)
original_counts = meta["dataset_stats"].get("target_counts_original", {})
total_n = int(len(df))
train_n = int(total_n * 0.70)
test_n = total_n - train_n

st.title("Student Dropout Risk Modeling for Early Intervention")
st.caption(
    "Dataset: UCI Machine Learning Repository - Predict Students' Dropout and Academic Success "
    "(DOI: 10.24432/C5MC89)"
)


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Predictive Analytics",
        "Explainability & Interactive Prediction",
    ]
)


with tab1:
    st.subheader("1.1 Dataset Introduction")
    st.markdown("### Dataset and Prediction Task")
    st.markdown(
        "**Overview:** This project uses machine learning to predict which university students are likely to drop out "
        "based on academic, financial, and demographic information. "
        "The dataset comes from the UCI Machine Learning Repository and was collected for a higher-education study in Portugal (2021), "
        "covering students across multiple Portuguese universities. "
        "The goal is to identify at-risk students early enough to support timely intervention."
    )
    st.markdown(
        "This project analyzes the UCI student outcomes dataset with **4,424 rows and 37 columns** "
        "(**36 predictors + 1 target**). "
        "According to UCI documentation, the records come from a higher education institution and were "
        "integrated from several disjoint databases that cover enrollment information and performance after the first two semesters. "
        "The original outcome is 3-class (`Dropout`, `Enrolled`, `Graduate`), and this HW1 implementation "
        "reformulates the task as binary: **Dropout = 1** and **Non-dropout (Enrolled + Graduate) = 0**."
    )
    st.markdown("UCI feature types include **real, integer, and categorical** variables.")
    st.markdown(
        "Source: [UCI Dataset Page](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)"
    )
    st.markdown(
        "**What the dataset contains:** demographic profile (e.g., age, gender, nationality), "
        "application/admission variables, academic progression variables (first/second semester performance), "
        "financial indicators (tuition status, debtor status, scholarship), and family-background attributes, "
        "including parents' academic background and occupation."
    )
    if original_counts:
        st.markdown(
            f"Original 3-class outcome distribution (before binary reformulation): "
            f"**Dropout = {original_counts.get('Dropout', 'N/A')}**, "
            f"**Enrolled = {original_counts.get('Enrolled', 'N/A')}**, "
            f"**Graduate = {original_counts.get('Graduate', 'N/A')}**."
        )

    st.subheader("Data Cleaning and Preparation Notes")
    st.markdown(
        "The raw CSV is semicolon-delimited, so the pipeline first standardizes column formatting and harmonizes types for modeling. "
        f"Data-quality checks on this submission dataset show **{missing_total} missing values** and **{duplicate_rows} duplicate rows**. "
        "Categorical-style fields were converted into model-ready numeric representations to match algorithm input requirements. "
        "For prediction, the original 3-class target is converted to a binary early-warning target focused on dropout risk."
    )
    st.markdown(
        "Original 3-class encoding used in exploratory steps: **Dropout = 0**, **Enrolled = 1**, **Graduate = 2**. "
        "Final HW1 prediction target for modeling and deployment: **Dropout = 1** vs **Non-dropout (Enrolled + Graduate) = 0**."
    )

    st.subheader("Why This Problem Matters")
    st.markdown(
        "Core goal: determine whether machine learning models can effectively identify students at risk of dropping out "
        "early enough to enable intervention and additional support."
    )
    st.markdown(
        "Student dropout is not only an academic KPI; it has direct implications for institutional planning, student support allocation, and equity outcomes. "
        "A reliable early-warning model helps identify high-risk students before failure compounds, enabling targeted interventions where they matter most."
    )
    st.markdown(
        "From an operations and business perspective, this supports advisor workload triage, retention-oriented scholarship policy, "
        "tuition-risk monitoring, and better allocation of limited student-success resources."
    )

    st.subheader("Approach and Key Findings")
    st.markdown(
        "The workflow follows full-stack data science practice: descriptive analytics, train-only feature recheck, multi-model tuning, explainability, and deployment. "
        "Model comparison shows that tree ensembles provide the strongest nonlinear predictive power, while logistic regression remains useful for interpretable directional effects."
    )
    st.markdown(
        f"In this run, overall dropout prevalence is **{eda_highlights['dropout_rate']:.1%}**. "
        f"The strongest early warning pattern is academic: average first-semester grade is **{eda_highlights['first_sem_dropout']:.2f}** in the dropout group "
        f"versus **{eda_highlights['first_sem_non_dropout']:.2f}** in the non-dropout group."
    )
    st.markdown(
        "The final app surfaces all required outputs: Part 1 visuals and interpretations, Part 2 metrics/ROC/hyperparameters, and Part 3 SHAP global + local explanations. "
        "Interactive prediction allows a user to set key feature values and inspect both predicted risk and feature-level contribution via SHAP waterfall."
    )

    ds = meta["dataset_stats"]
    st.subheader("Quick Dataset Stats")
    st.dataframe(
        pd.DataFrame(
            {
                "Rows": [ds["rows"]],
                "Columns": [ds["columns"]],
                "Dropout(1) Ratio": [round(ds["target_ratio_binary"]["1"], 4)],
                "Non-dropout(0) Ratio": [round(ds["target_ratio_binary"]["0"], 4)],
            }
        ),
        hide_index=True,
        width="stretch",
    )


with tab2:
    st.header("Part 1: Descriptive Analytics")

    st.subheader("1.2 Target Distribution")

    st.image(str(FIGURES / "part1_target_distribution.png"), width="stretch")
    st.caption(captions["target_distribution"])
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
        "How this imbalance is handled in this project (simple workflow):"
    )
    st.markdown(
        "1. **Keep class ratio stable in train/test** using stratified split.\n"
        "2. **Penalize mistakes on the dropout class more** using class weights.\n"
        "3. **Evaluate with F1 and AUC (not accuracy only)** so minority-class detection is reflected.\n"
        "4. **Review precision/recall trade-off** to match intervention policy."
    )

    st.subheader("1.3 Feature Distributions and Relationships")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(str(FIGURES / "part1_admission_grade_boxplot.png"), width="stretch")
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
        st.image(str(FIGURES / "part1_first_sem_grade_boxplot.png"), width="stretch")
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
    st.markdown("**Heatmap interpretation (5 practical takeaways):**")
    st.markdown(
        f"1. Academic continuity is strong across semesters: first-semester and second-semester grades move together "
        f"(**r = {corr_first_second_grade:.3f}**), and approved units show a similar pattern (**r = {corr_first_second_approved:.3f}**).\n"
        f"2. Parent occupation fields are highly aligned (**r = {corr_parent_occ:.3f}**), which suggests these columns capture overlapping background signal.\n"
        f"3. Second-semester grade has one of the strongest links with dropout (**r = {corr_target_second_grade:.3f}** with `Dropout_flag`): "
        "the negative sign means higher grades are associated with lower dropout risk, so low semester performance is an early warning sign.\n"
        f"4. Financial stress appears in the matrix too: tuition up-to-date is negatively related to dropout (**r = {corr_target_tuition:.3f}**), while debtor status is positively related (**r = {corr_target_debtor:.3f}**).\n"
        f"5. Student context matters beyond grades: scholarship status is protective (**r = {corr_target_scholarship:.3f}**) and age at enrollment shows added risk pressure (**r = {corr_target_age:.3f}**, positive sign)."
    )
    st.markdown("**What this means for modeling:**")
    st.markdown(
        "1. We prioritize academic-progress and financial-status variables as core predictors, because they carry the clearest risk signal in the correlation structure.\n"
        "2. We avoid feeding too many near-duplicate academic columns at once; train-only recheck and correlation filtering keep the feature set stable and reduce multicollinearity risk.\n"
        "3. We do not treat one high-correlation feature as the whole story. Final model choice still depends on out-of-sample F1/AUC and precision-recall trade-offs.\n"
        "4. Correlation is only pairwise association, so we validate interactions with tree-based models and use SHAP to confirm how features contribute to individual predictions.\n"
        "5. In practice, the strongest prediction signals here point to a combined intervention policy: monitor low semester performance together with tuition/debtor risk, then prioritize those students for early support."
    )


with tab3:
    st.header("Part 2: Predictive Analytics")

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
    best_row_by_f1 = comparison_df.sort_values("f1", ascending=False).iloc[0]
    best_model_name = str(best_row_by_f1["model"])
    best_model_f1 = float(best_row_by_f1["f1"])
    best_model_auc = float(best_row_by_f1["auc"])

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
        "**Preprocessing used (and why):**\n"
        "1. **Type handling/encoding:** predictors are loaded as numeric values; category-coded variables in this dataset are already integer-coded, so one-hot encoding is not required.\n"
        "2. **Missing values:** median imputation is applied to keep all rows and avoid dropping minority-risk cases.\n"
        "3. **Scaling:** `StandardScaler` is applied for Logistic Regression and MLP (scale-sensitive models); tree-based models use imputation only.\n"
        "4. **Leakage control:** feature recheck, CV tuning, and preprocessing fit happen on training data; test data is used only once for final evaluation."
    )
    with st.expander("2.1 Implementation Code (train_pipeline.py)"):
        st.code(
            "train_df, test_df = train_test_split(\n"
            "    df,\n"
            "    test_size=0.30,\n"
            "    stratify=df['Dropout_flag'],\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n\n"
            "X_train = train_df[final_features].copy()\n"
            "y_train = train_df['Dropout_flag'].astype(int).copy()\n"
            "X_test = test_df[final_features].copy()\n"
            "y_test = test_df['Dropout_flag'].astype(int).copy()\n\n"
            "logistic_preprocess = Pipeline(steps=[\n"
            "    ('imputer', SimpleImputer(strategy='median')),\n"
            "    ('scaler', StandardScaler()),\n"
            "])\n\n"
            "tree_preprocess = Pipeline(steps=[\n"
            "    ('imputer', SimpleImputer(strategy='median')),\n"
            "])",
            language="python",
        )
    st.markdown(
        f"**Final selected X features:** `{', '.join(feature_names)}`"
    )
    st.markdown("**Why `n=10` final features? (Train-only recheck evidence)**")
    st.markdown(
        "1. Start from the prior 18-feature baseline list.\n"
        "2. Recheck only on the training split to prevent leakage: univariate logistic screening (`p < 0.05`) and train-only correlation filtering (`|r| > 0.75`).\n"
        "3. Build a 10-feature candidate by ranking absolute logistic coefficients inside the rechecked 18-feature set.\n"
        "4. Compare `18-feature` vs `10-feature` sets using training-only 5-fold Stratified CV with F1 as the selection metric."
    )
    if not np.isnan(cv_f1_18) and not np.isnan(cv_f1_10):
        st.markdown(
            f"CV result: **F1(18) = {cv_f1_18:.3f}** vs **F1(10) = {cv_f1_10:.3f}** "
            f"(difference: **{cv_f1_10 - cv_f1_18:+.3f}** toward the selected set)."
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
        f"The binary target is imbalanced (**Dropout = {eda_highlights['dropout_rate']:.1%}**, "
        f"**Non-dropout = {(1-eda_highlights['dropout_rate']):.1%}**). "
        "Our handling strategy is practical and model-consistent: "
        "(1) **stratified split** preserves the same class ratio in train and test; "
        "(2) **class-weighted learning** penalizes dropout-class errors more (`class_weight='balanced'` for logistic/tree/forest/LightGBM, and computed class weights for MLP); "
        "(3) **selection by F1 and AUC** keeps the focus on minority-class detection quality instead of accuracy-only ranking; "
        "(4) **precision/recall trade-off review** is reported for each model so intervention policy can choose conservative vs aggressive alerting."
    )
    st.markdown(
        "We intentionally did not rely on random oversampling in this submission, because weighted objectives already improved minority sensitivity while keeping the pipeline simple and reproducible."
    )

    st.subheader("2.2 Logistic Regression Baseline")
    st.markdown(
        "Classification baseline model is Logistic Regression. "
        "Required test metrics (Accuracy, Precision, Recall, F1, AUC-ROC) are reported below."
    )
    st.dataframe(one_row_metrics("logistic", "Logistic Regression (Baseline)"), hide_index=True, width="stretch")
    st.image(str(FIGURES / "part2_roc_logistic.png"), width="stretch")
    st.markdown(
        f"Result interpretation: Logistic baseline reaches **Accuracy {baseline_accuracy:.3f}**, **Precision {baseline_precision:.3f}**, "
        f"**Recall {baseline_recall:.3f}**, **F1 {baseline_f1:.3f}**, and **AUC {baseline_auc:.3f}** on the held-out test set."
    )
    st.markdown(
        "This is a strong baseline for this dataset: recall is relatively high, so the model catches many at-risk students, "
        "while AUC above 0.90 indicates good ranking ability across thresholds. "
        "Precision is lower than recall, which means the model is intentionally more sensitive (more alerts, including some false positives)."
    )
    if not np.isnan(baseline_f1):
        st.markdown(
            f"Baseline role: every later model is judged against this reference. "
            f"Current best F1 model is **{best_model_name}** with **F1 {best_model_f1:.3f}** "
            f"(improvement **{best_model_f1 - baseline_f1:+.3f}** over Logistic) and **AUC {best_model_auc:.3f}** "
            f"(delta **{best_model_auc - baseline_auc:+.3f}**)."
        )

    st.subheader("2.3 Decision Tree (GridSearchCV, 5-fold)")
    st.markdown(
        "Tuning grid: `max_depth = [3, 5, 7, 10]`, `min_samples_leaf = [5, 10, 20, 50]`, `scoring = F1`."
    )
    st.markdown(f"Best params: `{best_params.get('decision_tree', {})}`")
    st.dataframe(one_row_metrics("decision_tree", "Decision Tree"), hide_index=True, width="stretch")
    tree_col1, tree_col2 = st.columns(2)
    with tree_col1:
        st.image(str(FIGURES / "part2_best_decision_tree.png"), width="stretch")
    with tree_col2:
        st.image(str(FIGURES / "part2_roc_decision_tree.png"), width="stretch")

    st.subheader("2.4 Random Forest (GridSearchCV, 5-fold)")
    st.markdown(
        "Tuning grid: `n_estimators = [50, 100, 200]`, `max_depth = [3, 5, 8]`, `scoring = F1`."
    )
    st.markdown(f"Best params: `{best_params.get('random_forest', {})}`")
    st.dataframe(one_row_metrics("random_forest", "Random Forest"), hide_index=True, width="stretch")
    st.image(str(FIGURES / "part2_roc_random_forest.png"), width="stretch")

    st.subheader("2.5 Boosted Tree (LightGBM, GridSearchCV, 5-fold)")
    st.markdown(
        "Boosted-tree implementation uses LightGBM. "
        "Tuning covers at least 3 hyperparameters: "
        "`n_estimators = [50, 100, 200]`, `max_depth = [3, 4, 5, 6]`, `learning_rate = [0.01, 0.05, 0.1]` with `scoring = F1`."
    )
    st.markdown(f"Best params: `{best_params.get('lightgbm', {})}`")
    st.dataframe(one_row_metrics("lightgbm", "Boosted Tree (LightGBM)"), hide_index=True, width="stretch")
    st.image(str(FIGURES / "part2_roc_lightgbm.png"), width="stretch")

    st.subheader("2.6 Neural Network (Keras MLP)")
    st.markdown(
        "Architecture: input layer -> at least two hidden ReLU layers -> sigmoid output. "
        "Loss: `binary_crossentropy`; Optimizer: `Adam`; class weights applied for imbalance handling."
    )
    st.markdown(f"Best tuned MLP config: `{best_params.get('mlp_keras', {})}`")
    st.dataframe(one_row_metrics("mlp_keras", "MLP (Keras)"), hide_index=True, width="stretch")
    mlp_col1, mlp_col2 = st.columns(2)
    with mlp_col1:
        st.image(str(FIGURES / "part2_mlp_training_history.png"), width="stretch")
    with mlp_col2:
        st.image(str(FIGURES / "part2_roc_mlp_keras.png"), width="stretch")

    st.subheader("Bonus +1: MLP Hyperparameter Tuning")
    st.image(str(FIGURES / "bonus_mlp_tuning_heatmap.png"), width="stretch")

    st.subheader("2.7 Model Comparison Summary")
    st.dataframe(comparison_df, width="stretch")
    st.image(str(FIGURES / "part2_f1_bar_comparison.png"), width="stretch")
    if not np.isnan(baseline_f1):
        delta = best_model_f1 - baseline_f1
        if best_model_name != "logistic":
            st.markdown(
                f"Baseline check: Logistic F1 is **{baseline_f1:.3f}**. "
                f"Best model by F1 is **{best_model_name}** at **{best_model_f1:.3f}** "
                f"(improvement **{delta:.3f}** over baseline)."
            )
        else:
            st.markdown(
                f"Baseline check: Logistic is currently the top F1 model at **{baseline_f1:.3f}**."
            )

    st.subheader("Best Hyperparameters (All Models)")
    st.json(best_params)

    st.subheader("ROC Curves (All Models)")
    roc_cols = st.columns(2)
    roc_files = [
        "part2_roc_logistic.png",
        "part2_roc_decision_tree.png",
        "part2_roc_random_forest.png",
        "part2_roc_lightgbm.png",
        "part2_roc_mlp_keras.png",
    ]
    for i, fn in enumerate(roc_files):
        with roc_cols[i % 2]:
            st.image(str(FIGURES / fn), width="stretch")

    st.markdown(tradeoff_text)


with tab4:
    st.header("Part 3: SHAP Explainability")

    model_label = {
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "lightgbm": "LightGBM",
        "logistic": "Logistic Regression",
        "mlp_keras": "MLP (Keras)",
    }

    st.subheader("3.1 Best-performing Tree-based Model Used for SHAP")
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
    st.dataframe(tree_perf_view, hide_index=True, width="stretch")

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

    st.subheader("3.2 Global SHAP Plots")
    st.image(str(FIGURES / "part3_shap_summary_beeswarm.png"), width="stretch")
    st.image(str(FIGURES / "part3_shap_bar.png"), width="stretch")

    shap_rank_df = pd.DataFrame(shap_interpretation)
    if not shap_rank_df.empty:
        shap_rank_df = shap_rank_df.copy()
        shap_rank_df["Rank"] = np.arange(1, len(shap_rank_df) + 1)
        shap_rank_df["mean|SHAP|"] = shap_rank_df["mean_abs_shap"]
        shap_rank_df["Direction"] = shap_rank_df["direction"]
        st.dataframe(
            shap_rank_df[["Rank", "feature", "mean|SHAP|", "Direction"]].rename(columns={"feature": "Feature"}),
            hide_index=True,
            width="stretch",
        )

    st.subheader("3.3 Required Interpretation")
    top_feature_names = ", ".join([f"`{x['feature']}`" for x in shap_interpretation])
    direction_sentence = "; ".join([f"`{x['feature']}` -> {x['direction']}" for x in shap_interpretation])
    st.markdown(f"1. **Strongest impact features:** {top_feature_names}.")
    st.markdown(f"2. **Direction of influence:** {direction_sentence}.")
    st.markdown(
        "3. **Decision-use implication:** SHAP output supports operational triage: prioritize students when multiple risk drivers are active together "
        "(low academic progression, tuition/debtor risk, and vulnerable profile factors), then align interventions by driver type."
    )

    extra_insights: List[str] = []
    shap_feature_set = {x["feature"] for x in shap_interpretation}
    if "Curricular units 2nd sem (approved)" in shap_feature_set:
        extra_insights.append(
            "Academic progression dominates risk separation: when approved units in semester 2 are low, SHAP contributions frequently shift predictions toward dropout."
        )
    if "Tuition fees up to date" in shap_feature_set:
        extra_insights.append(
            "Tuition payment status behaves as a strong protective/risk switch: up-to-date payment tends to push risk down, while non-payment pushes it up."
        )
    if "Curricular units 1st sem (grade)" in shap_feature_set:
        extra_insights.append(
            "Early academic signal persists even after adding other variables: first-semester grade still contributes meaningful incremental information."
        )
    if "Age at enrollment" in shap_feature_set:
        extra_insights.append(
            "Age effects are directional but not deterministic: older enrollment tends to increase predicted risk on average, but SHAP shows overlap across individuals."
        )
    if "Scholarship holder" in shap_feature_set:
        extra_insights.append(
            "Scholarship status appears as a retention buffer in the model: scholarship holder values generally pull risk downward."
        )
    if extra_insights:
        st.markdown("**Additional SHAP insights:** " + " ".join(extra_insights))

    st.subheader("3.4 Interactive Prediction + Custom SHAP Waterfall")
    model_options = ["logistic", "decision_tree", "random_forest", "lightgbm", "mlp_keras"]
    default_idx = model_options.index(best_tree_model) if best_tree_model in model_options else 2
    selected_model = st.selectbox("Select model for prediction", model_options, index=default_idx)

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

    if st.button("Run Prediction", type="primary"):
        prob, pred = predict_with_model(selected_model, row_df, feature_names, models)
        st.success(
            f"Predicted class: **{'Dropout (1)' if pred == 1 else 'Non-dropout (0)'}**  |  "
            f"Predicted probability of dropout: **{prob:.4f}**"
        )

        if selected_model in {"decision_tree", "random_forest", "lightgbm"}:
            shap_model = selected_model
            st.info(
                f"Waterfall explanation uses the selected tree model: **{model_label.get(selected_model, selected_model)}**."
            )
        else:
            shap_model = best_tree_model
            st.info(
                f"Selected model **{model_label.get(selected_model, selected_model)}** is non-tree. "
                f"For local SHAP explainability, waterfall is generated with the best tree-based model "
                f"**{model_label.get(best_tree_model, best_tree_model)}**."
            )

        fig = make_custom_waterfall(shap_model, row_df, feature_names, models)
        st.pyplot(fig, clear_figure=True)

    st.subheader("3.5 Reference Waterfall Example from Test Set")
    st.image(str(FIGURES / "part3_shap_waterfall_example.png"), width="stretch")
