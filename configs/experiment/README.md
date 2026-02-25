# 🧪 Эксперименты на ClearML

Всего проведено **29 экспериментов** с полным логированием в ClearML.  
Каждый эксперимент содержит:
- `ROC/PR` кривые
- `Confusion matrix`
- Распределение предсказаний
- `Feature importance` / анализ коэффициентов линейной модели
- Результаты `GridSearchCV` (где применимо)
- Все метрики (`AUC`, `AP`, `Precision`, `Recall`, `F1`, `MCC (Matthews Correlation Coefficient)`)

## 📊 Logistic Regression (4 эксперимента)

| № | Эксперимент | AUC | Ссылка |
|---|-------------|-----|--------|
| 1 | logreg_lbfgs_l2 | 0.861843 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/cd326b00d1b74dd18fad4c7d904ee974/output/execution) |
| 2 | logreg_cv_liblinear_l2 | 0.862310| [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/4b47451ff78b4b99b827ece2bfea03c4/output/execution) |
| 3 | logreg_cv_saga_l2 | 0.861719 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/31f5d7c550a34f1f87a516f2dfa7de5b/output/execution) |
| 4 | logreg_final (C=550, l1) | **0.85804** | [🔗 ClearML](ССЫЛКА_4) |

## 📊 Decision Tree (3 эксперимента)

| № | Эксперимент | AUC | Ссылка |
|---|-------------|-----|--------|
| 1 | dtc_baseline | 0.8451 | [🔗 ClearML](ССЫЛКА_5) |
| 2 | dtc_gridsearch | 0.8512 | [🔗 ClearML](ССЫЛКА_6) |
| 3 | dtc_final | **0.85330** | [🔗 ClearML](ССЫЛКА_7) |

## 📊 Random Forest (4 эксперимента)

| № | Эксперимент | AUC | Ссылка |
|---|-------------|-----|--------|
| 1 | rfc_baseline | 0.8589 | [🔗 ClearML](ССЫЛКА_8) |
| 2 | rfc_gridsearch_v1 | 0.8601 | [🔗 ClearML](ССЫЛКА_9) |
| 3 | rfc_gridsearch_v2 | 0.8615 | [🔗 ClearML](ССЫЛКА_10) |
| 4 | rfc_final | **0.86341** | [🔗 ClearML](ССЫЛКА_11) |

## 📊 LightGBM (5 экспериментов)

| № | Эксперимент | AUC | Ссылка |
|---|-------------|-----|--------|
| 1 | lgbm_baseline | 0.8631 | [🔗 ClearML](ССЫЛКА_12) |
| 2 | lgbm_randomsearch_v1 | 0.8652 | [🔗 ClearML](ССЫЛКА_13) |
| 3 | lgbm_randomsearch_v2 | 0.8661 | [🔗 ClearML](ССЫЛКА_14) |
| 4 | lgbm_tuned | 0.8668 | [🔗 ClearML](ССЫЛКА_15) |
| 5 | lgbm_final | **0.86705** | [🔗 ClearML](ССЫЛКА_16) |

## 📊 CatBoost (4 эксперимента)

| № | Эксперимент | AUC | Ссылка |
|---|-------------|-----|--------|
| 1 | catboost_baseline | 0.8645 | [🔗 ClearML](ССЫЛКА_17) |
| 2 | catboost_randomsearch_v1 | 0.8662 | [🔗 ClearML](ССЫЛКА_18) |
| 3 | catboost_randomsearch_v2 | 0.8670 | [🔗 ClearML](ССЫЛКА_19) |
| 4 | catboost_final | **0.86695** | [🔗 ClearML](ССЫЛКА_20) |

## 📊 XGBoost (6 экспериментов)

| № | Эксперимент | AUC | Ссылка |
|---|-------------|-----|--------|
| 1 | xgb_baseline | 0.8648 | [🔗 ClearML](ССЫЛКА_21) |
| 2 | xgb_randomsearch_v1 | 0.8665 | [🔗 ClearML](ССЫЛКА_22) |
| 3 | xgb_randomsearch_v2 | 0.8673 | [🔗 ClearML](ССЫЛКА_23) |
| 4 | xgb_n_estimators_500 | 0.8675 | [🔗 ClearML](ССЫЛКА_24) |
| 5 | xgb_n_estimators_1000 | 0.8678 | [🔗 ClearML](ССЫЛКА_25) |
| 6 | xgb_final | **0.86672** | [🔗 ClearML](ССЫЛКА_26) |

## 📊 Ансамбли (3 эксперимента)

| № | Эксперимент | AUC | Ссылка |
|---|-------------|-----|--------|
| 1 | ensemble_xgb_rf_7_3 | 0.8683 | [🔗 ClearML](ССЫЛКА_27) |
| 2 | ensemble_xgb_rf_lr | 0.8682 | [🔗 ClearML](ССЫЛКА_28) |
| 3 | ensemble_weighted | **0.8684** | [🔗 ClearML](ССЫЛКА_29) |

## 🏆 Итоговая таблица лучших моделей

| Место | Модель | AUC |
|-------|--------|-----|
| 🥇 | Weighted Ensemble | **0.8684** |
| 🥈 | LightGBM | 0.86705 |
| 🥉 | CatBoost | 0.86695 |
| 4 | XGBoost | 0.86672 |
| 5 | Random Forest | 0.86341 |
| 6 | Logistic Regression | 0.85804 |
| 7 | Decision Tree | 0.85330 |

## 🔗 Прямые ссылки на лучшие эксперименты

- [LightGBM final](ССЫЛКА_16)
- [XGBoost final](ССЫЛКА_26)
- [CatBoost final](ССЫЛКА_20)
- [Logistic Regression final](ССЫЛКА_4)
- [Weighted Ensemble](ССЫЛКА_29)

## 📈 Что логировалось в каждом эксперименте

Каждый эксперимент в ClearML содержит:
- **Scalars:** AUC, AP, Precision, Recall, F1, MCC
- **Plots:** ROC curve, PR curve, confusion matrix, distribution plot
- **Feature importance** (для tree-based)
- **Coefficients** (для линейных моделей)
- **GridSearchCV results** (где применимо)
- **Hyperparameters** (полный конфиг эксперимента)
- **Artifacts:** модель, препроцессор, скейлер, submission.csv

## 📚 Подробнее

- [Обучение моделей](src/README.md)
- [Конфиги Hydra](configs/README.md)
- [Streamlit приложение](app/README.md)
