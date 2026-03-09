# MSIS 522 HW1 - Data Science Workflow (Dropout Dataset)

This repository implements the full HW1 workflow using the UCI student dropout dataset.
It follows the required structure: descriptive analytics, predictive modeling, SHAP explainability, and Streamlit deployment.

## 1. Dataset
- File: `data/studentdata_raw.csv`
- Source: [UCI Machine Learning Repository - Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- DOI: `10.24432/C5MC89`
- Local data shape: 4,424 rows and 37 columns (`36 predictors + 1 target`)
- Feature types (UCI): real, integer, and categorical variables
- Problem type: Classification (binary formulation for HW1: `Dropout=1`, `Enrolled+Graduate=0`)
- Data context (UCI): records from a higher education institution, merged from several disjoint databases at enrollment and after first/second semester performance.

## 2. Dataset Description for Report/Slides
- Overview: This project predicts which university students are likely to drop out using academic, financial, and demographic data.
- Source context: UCI Machine Learning Repository; records were collected for a higher-education study in Portugal (2021), including students across multiple Portuguese universities.
- Goal: test whether machine learning can identify at-risk students early enough to support timely intervention.
- The dataset includes **4,424 students** and **36 predictive features** covering demographic, application, financial, and academic dimensions.
- Example feature groups: age/gender/nationality, admission profile, first/second-semester grades and approvals, tuition/debtor/scholarship indicators, and family background (including parents' education and occupation).
- The original UCI task is a 3-class outcome (`Dropout`, `Enrolled`, `Graduate`) with class imbalance; this HW1 implementation reformulates the target into binary dropout risk.
- Original exploratory 3-class encoding reference: `Dropout=0`, `Enrolled=1`, `Graduate=2`.
- Target used in this repository:
  - `1 = Dropout`
  - `0 = Enrolled or Graduate`
- Data cleaning checks on the submitted dataset: 0 missing values, 0 duplicated rows.

## 3. HW1 Requirement Coverage
- Train/test split first (`70/30`, `stratify`, `random_state=42`)
- Train-only feature recheck (p-value screening + correlation filtering)
- 5-fold `GridSearchCV` for Decision Tree, Random Forest, and Boosted Trees
- Keras MLP and bonus hyperparameter tuning
- Imbalanced target handling:
  - stratified split
  - class-weighted training (`class_weight='balanced'` for tree/logistic models, computed class weights for MLP)
  - F1 and AUC emphasized for model selection
- SHAP required plots (beeswarm, bar, waterfall)
- Streamlit app with 4 required tabs and interactive prediction

### 2.1 Data Preparation (Required)
- **Target (`y`)**: `Dropout_flag`
  - `1 = Dropout`
  - `0 = Non-dropout (Enrolled + Graduate)`
- **Features (`X`)**: train-only rechecked final feature set (saved in `artifacts/metadata/project_metadata.json` under `feature_selection.final_features`).
- **Split**: `train_test_split(..., test_size=0.30, stratify=y, random_state=42)`.
- **Preprocessing**:
  - Numeric conversion on load (`pd.to_numeric(..., errors='coerce')`).
  - Median imputation for missing values (`SimpleImputer(strategy='median')`).
  - `StandardScaler` for Logistic Regression and MLP only.
  - Tree-based models (Decision Tree / Random Forest / LightGBM) use imputation without scaling.
- **Why this setup**:
  - Stratified split preserves class ratio.
  - Median imputation is robust to skew and avoids dropping data.
  - Scaling is applied only where model optimization depends on feature scale.
  - All feature recheck/tuning/preprocessing fit steps are done on training data to prevent leakage.

## 4. Train Pipeline
Main script: `train_pipeline.py`

### Run
```bash
python3 train_pipeline.py
```

### What it does
1. Loads/cleans data.
2. Splits train/test first (`70/30`, `stratify`, `random_state=42`).
3. Rechecks selected features on train-only:
   - univariate logistic screen (`p < 0.05`)
   - correlation filtering (`|r| > 0.75`)
   - creates 18-feature set
   - creates 10-feature set and compares CV F1 (`5-fold`)
   - chooses final feature set by higher CV F1 (tie -> 18 features)
4. Trains/evaluates required models:
   - Logistic baseline
   - Decision Tree (GridSearchCV)
   - Random Forest (GridSearchCV)
   - LightGBM boosted tree (GridSearchCV, 3+ hyperparameters)
   - Keras MLP
5. Bonus: MLP hyperparameter tuning + heatmap visualization.
6. SHAP on best tree-based model:
   - summary beeswarm
   - mean |SHAP| bar
   - waterfall for one high-risk case
7. Saves models, metrics, params, and figures for Streamlit.

## 5. Streamlit App
Main app: `app.py`

### Run
```bash
streamlit run app.py
```

### Required tabs included
- **Tab 1 - Executive Summary**
- **Tab 2 - Descriptive Analytics**
- **Tab 3 - Model Performance**
- **Tab 4 - Explainability & Interactive Prediction**

### Interactive section
- User can choose prediction model.
- User can set key feature values via widgets.
- App returns class + dropout probability.
- SHAP waterfall is shown for selected tree model (or best tree fallback for non-tree models).

## 6. Artifacts
Generated under `artifacts/`:
- `models/`: saved pretrained models (no retraining in app)
- `metrics/`: model metrics, best params, comparison tables
- `figures/`: Part 1/2/3 and bonus plots
- `metadata/`: project metadata and app content references

## 7. HW1 Deliverables Checklist
- [x] Analysis code (scripts)
- [x] Streamlit app code (`app.py`)
- [x] Saved model files
- [x] `requirements.txt`
- [x] `runtime.txt` (Streamlit Cloud Python version pin)
- [x] `README.md`
- [x] Public deployed Streamlit URL: [https://studentdropout-prediction-ml.streamlit.app/](https://studentdropout-prediction-ml.streamlit.app/)

## 8. Reproducibility Notes
- `random_state=42` is used for all stochastic operations.
- App loads saved models; it does not retrain models on the fly.
- If running on macOS Apple Silicon, TensorFlow install may differ from Linux cloud runtime.
