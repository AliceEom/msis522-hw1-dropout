#!/usr/bin/env python3
from __future__ import annotations

import json
import os
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

    out["admission_non_dropout"] = float(tmp.loc[tmp["Dropout_flag"] == 0, "Admission grade"].mean())
    out["admission_dropout"] = float(tmp.loc[tmp["Dropout_flag"] == 1, "Admission grade"].mean())

    out["first_sem_non_dropout"] = float(tmp.loc[tmp["Dropout_flag"] == 0, "Curricular units 1st sem (grade)"].mean())
    out["first_sem_dropout"] = float(tmp.loc[tmp["Dropout_flag"] == 1, "Curricular units 1st sem (grade)"].mean())

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
    top_idx = np.argsort(mean_abs)[::-1][:3]
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
manual_input_features = meta.get("manual_input_features", [])
feature_ranges = meta["feature_ranges"]
feature_means = meta["feature_means"]
best_tree_model = meta["best_tree_model_for_shap"]
captions = meta["captions"]
top_corr_a, top_corr_b, top_corr_val = strongest_corr_pair(df, feature_names)
shap_interpretation = compute_shap_interpretation(df, feature_names, best_tree_model)
eda_highlights = compute_eda_highlights(df)
tradeoff_text = build_model_tradeoff_text(comparison_df)
original_counts = meta["dataset_stats"].get("target_counts_original", {})

st.title("Student Dropout Risk Modeling for Early Intervention")
st.caption(
    "Dataset: UCI Machine Learning Repository - Predict Students' Dropout and Academic Success "
    "(DOI: 10.24432/C5MC89)"
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
    st.subheader("Dataset and Prediction Task")
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
    st.subheader("1.1 Dataset Introduction")
    st.markdown(
        "Dataset context, target definition, feature types, and data-cleaning summary are provided in **Tab 1 (Executive Summary)** "
        "to match the assignment requirement for a clear dataset introduction."
    )

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
            f"**{eda_highlights['admission_non_dropout_median']:.2f}** (non-dropout), and IQR is "
            f"**{eda_highlights['admission_dropout_iqr']:.2f}** vs **{eda_highlights['admission_non_dropout_iqr']:.2f}**. "
            f"Using the 1.5×IQR rule, outlier rates are **{eda_highlights['admission_dropout_outlier_rate']:.1%}** "
            f"(dropout) and **{eda_highlights['admission_non_dropout_outlier_rate']:.1%}** (non-dropout), "
            "which indicates overlap but still a meaningful central-shift toward lower admission readiness in the dropout group."
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
            f"**{eda_highlights['first_sem_non_dropout_median']:.2f}** (non-dropout), with IQR "
            f"**{eda_highlights['first_sem_dropout_iqr']:.2f}** vs **{eda_highlights['first_sem_non_dropout_iqr']:.2f}**. "
            f"Outlier rates by the 1.5×IQR rule are **{eda_highlights['first_sem_dropout_outlier_rate']:.1%}** "
            f"and **{eda_highlights['first_sem_non_dropout_outlier_rate']:.1%}** respectively. "
            "The larger center gap than in admission scores suggests first-semester performance is a much sharper early-separation signal."
        )

    st.image(str(FIGURES / "part1_dropout_by_grade_quartile.png"), width="stretch")
    st.caption(captions["dropout_by_grade_quartile"])
    st.markdown(
        f"Dropout risk by quartile is highly nonlinear: **Q1 = {eda_highlights['dropout_q1']:.1%}** vs **Q4 = {eda_highlights['dropout_q4']:.1%}**. "
        "In practical terms, students in the lowest performance quartile should be prioritized for early support."
    )
    st.markdown(
        f"Financial signals show a similar risk pattern: debtors have **{eda_highlights['dropout_debtor_1']:.1%}** dropout "
        f"vs **{eda_highlights['dropout_debtor_0']:.1%}** for non-debtors; scholarship holders have **{eda_highlights['dropout_scholar_1']:.1%}** "
        f"vs **{eda_highlights['dropout_scholar_0']:.1%}** for non-holders."
    )

    st.subheader("Additional Relationship Views")
    col_c, col_d = st.columns(2)
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

    st.subheader("1.4 Correlation Heatmap")
    st.image(str(FIGURES / "part1_correlation_heatmap.png"), width="stretch")
    st.caption(captions["correlation_heatmap"])
    if not np.isnan(top_corr_val):
        st.markdown(
            f"Strongest absolute correlation in the selected feature set: **{top_corr_a}** vs **{top_corr_b}** "
            f"with **|r| = {top_corr_val:.3f}**."
        )


with tab3:
    st.subheader("2.1 Data Preparation")
    st.markdown(
        "Target is encoded as `Dropout_flag` (1/0), then data is split with a stratified 70/30 train-test split (`random_state=42`). "
        "Missing values are imputed; scaling is applied where needed (logistic/MLP), and all feature recheck and tuning decisions are performed on training data to prevent leakage."
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

    st.subheader("2.7 Model Comparison Summary")
    st.dataframe(comparison_df, width="stretch")

    st.image(str(FIGURES / "part2_f1_bar_comparison.png"), width="stretch")

    st.subheader("Best Hyperparameters")
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

    st.subheader("Decision Tree Snapshot")
    st.image(str(FIGURES / "part2_best_decision_tree.png"), width="stretch")

    st.subheader("MLP Training History")
    st.image(str(FIGURES / "part2_mlp_training_history.png"), width="stretch")

    st.subheader("Bonus: MLP Tuning Visualization")
    st.image(str(FIGURES / "bonus_mlp_tuning_heatmap.png"), width="stretch")

    st.markdown(tradeoff_text)


with tab4:
    st.subheader("SHAP Global Explanations")
    st.image(str(FIGURES / "part3_shap_summary_beeswarm.png"), width="stretch")
    st.image(str(FIGURES / "part3_shap_bar.png"), width="stretch")

    st.markdown(
        "**Interpretation guidance:** Features at the top of the SHAP ranking have the strongest average impact on risk scores. "
        "Positive SHAP values push toward dropout, while negative values push toward non-dropout."
    )
    st.subheader("Required Interpretation")
    st.markdown(
        "1. **Strongest impact features:** "
        + ", ".join([f"`{x['feature']}`" for x in shap_interpretation])
        + "."
    )
    st.markdown(
        "2. **Direction of influence:** "
        + "; ".join([f"`{x['feature']}`: {x['direction']}" for x in shap_interpretation])
        + "."
    )
    st.markdown(
        "3. **Decision-use implication:** These drivers can support proactive intervention policies (academic support, advising, and tuition-risk outreach) by identifying which factors most strongly move individual risk predictions."
    )

    st.subheader("Interactive Prediction")
    model_options = ["logistic", "decision_tree", "random_forest", "lightgbm", "mlp_keras"]
    selected_model = st.selectbox("Select model for prediction", model_options, index=2)

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
            st.info(f"SHAP waterfall is generated using the selected model: `{selected_model}`.")
        else:
            shap_model = best_tree_model
            st.info(
                f"Selected model `{selected_model}` is non-tree. For local explainability, waterfall is shown with best tree-based model `{best_tree_model}`."
            )

        fig = make_custom_waterfall(shap_model, row_df, feature_names, models)
        st.pyplot(fig, clear_figure=True)

    st.subheader("Reference Example from Test Set")
    st.image(str(FIGURES / "part3_shap_waterfall_example.png"), width="stretch")
