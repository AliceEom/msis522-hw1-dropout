# MSIS 522 HW1 - Data Science Workflow (Dropout Dataset)

This repository implements the full HW1 workflow using the UCI student dropout dataset.
It follows the required structure: descriptive analytics, predictive modeling, SHAP explainability, and Streamlit deployment.

## 1. Dataset
- File: `data/studentdata_raw.csv`
- Source: UCI Student Dropout and Academic Success dataset
- Rows: 4,424
- Columns: 37
- Task formulation: Binary classification (`Dropout=1`, `Enrolled+Graduate=0`)

## 2. Reuse Strategy (from previous project)
- Prior notebook and PPT insights were reused for:
  - EDA interpretation style/captions
  - feature-selection rationale
  - model trade-off narrative
- Prior source files were used locally for writing style and interpretation consistency.
- Legacy files are intentionally excluded from this submission repo to minimize unnecessary personal/project history exposure.
- New implementation adds HW1-specific requirements that were missing previously:
  - train-only recheck for feature selection
  - 5-fold GridSearchCV for CART/RF/Boosted Trees
  - Keras MLP + bonus hyperparameter tuning
  - SHAP required 3 plots
  - Streamlit 4-tab app with interactive prediction

## 3. Train Pipeline
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

## 4. Streamlit App
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

## 5. Artifacts
Generated under `artifacts/`:
- `models/`: saved pretrained models (no retraining in app)
- `metrics/`: model metrics, best params, comparison tables
- `figures/`: Part 1/2/3 and bonus plots
- `metadata/`: project metadata and app content references

## 6. HW1 Deliverables Checklist
- [x] Analysis code (scripts)
- [x] Streamlit app code (`app.py`)
- [x] Saved model files
- [x] `requirements.txt`
- [x] `runtime.txt` (Streamlit Cloud Python version pin)
- [x] `README.md`
- [ ] Public deployed Streamlit URL (to add after deployment)

## 7. Reproducibility Notes
- `random_state=42` is used for all stochastic operations.
- App loads saved models; it does not retrain models on the fly.
- If running on macOS Apple Silicon, TensorFlow install may differ from Linux cloud runtime.
