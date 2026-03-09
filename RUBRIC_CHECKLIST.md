# MSIS 522 HW1 Rubric Checklist (Requirement -> Evidence)

## Dataset & Setup
- [x] Tabular dataset with 100+ rows
  - Evidence: `data/studentdata_raw.csv` (4,424 rows), `artifacts/metadata/project_metadata.json`
- [x] Clear prediction target
  - Evidence: binary `Dropout_flag` in `train_pipeline.py`, summary in `app.py` Tab 1
- [x] Imbalance addressed
  - Evidence: F1-first comparison + class_weight usage in models (`train_pipeline.py`)

## Part 1 (25)
- [x] 1.1 Dataset introduction
  - Evidence: `app.py` Tab 1 text + stats table
- [x] 1.2 Target distribution plot + interpretation
  - Evidence: `artifacts/figures/part1_target_distribution.png`, caption in metadata/app
- [x] 1.3 >=4 insightful visualizations + interpretation
  - Evidence:
    - `part1_admission_grade_boxplot.png`
    - `part1_first_sem_grade_boxplot.png`
    - `part1_dropout_by_grade_quartile.png`
    - `part1_target_distribution.png`
    - Captions in `project_metadata.json` + Tab 2
- [x] 1.4 Correlation heatmap + strongest-correlation comment
  - Evidence: `part1_correlation_heatmap.png`, caption in Tab 2

## Part 2 (45)
- [x] 2.1 Data prep + split + preprocessing documentation
  - Evidence: `train_pipeline.py` (70/30 split first, train-only feature recheck)
- [x] 2.2 Logistic baseline + Acc/Prec/Rec/F1/AUC
  - Evidence: `part2_metrics.json` -> `logistic`
- [x] 2.3 Decision Tree with 5-fold GridSearchCV
  - Evidence: `train_pipeline.py` grid settings, `part2_best_params.json`, `part2_best_decision_tree.png`
- [x] 2.4 Random Forest with 5-fold GridSearchCV
  - Evidence: `train_pipeline.py` grid settings, `part2_best_params.json`, `part2_roc_random_forest.png`
- [x] 2.5 Boosted Trees with 5-fold GridSearchCV + >=3 hyperparameters
  - Evidence: LightGBM grid (`n_estimators`, `max_depth`, `learning_rate`), `part2_roc_lightgbm.png`
- [x] 2.6 Neural Network (Keras MLP)
  - Evidence: Keras model in `train_pipeline.py`, saved model `mlp_keras_model.keras`, `part2_mlp_training_history.png`
- [x] 2.7 Comparison table + key metric bar chart + trade-off paragraph
  - Evidence: `part2_model_comparison.csv`, `part2_f1_bar_comparison.png`, `model_comparison_paragraph.txt`, Tab 3

## Part 3 (10)
- [x] SHAP summary beeswarm
  - Evidence: `part3_shap_summary_beeswarm.png`
- [x] SHAP mean |value| bar
  - Evidence: `part3_shap_bar.png`
- [x] SHAP waterfall for one prediction
  - Evidence: `part3_shap_waterfall_example.png`
- [x] Interpretation of strongest features and direction
  - Evidence: Tab 4 explanatory text + plots

## Part 4 Streamlit (20)
- [x] 4 required tabs using `st.tabs`
  - Evidence: `app.py`
- [x] Tab 1 executive summary with problem meaning and approach
  - Evidence: `app.py` Tab 1
- [x] Tab 2 descriptive visuals + captions
  - Evidence: `app.py` Tab 2
- [x] Tab 3 model performance, metrics, ROC, best hyperparameters
  - Evidence: `app.py` Tab 3
- [x] Tab 4 SHAP + interactive prediction + model selection + probability + waterfall
  - Evidence: `app.py` Tab 4

## Bonus (+1)
- [x] MLP hyperparameter tuning + visualization
  - Evidence: `bonus_mlp_tuning_results.csv`, `bonus_mlp_tuning_heatmap.png`

## Deliverables
- [x] Analysis code
  - Evidence: `train_pipeline.py`
- [x] Streamlit app code
  - Evidence: `app.py`
- [x] Saved model files
  - Evidence: `artifacts/models/*`
- [x] Requirements file
  - Evidence: `requirements.txt`
- [x] README
  - Evidence: `README.md`
- [ ] Public deployed link (to be added after deployment)
