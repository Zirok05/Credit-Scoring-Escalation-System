# 🧪 Эксперименты на ClearML

Всего проведено **27 экспериментов** с полным логированием в ClearML.  
Каждый эксперимент содержит:
- `ROC/PR` кривые
- `Confusion matrix`
- Распределение предсказаний
- `Feature importance` / анализ коэффициентов линейной модели
- Результаты `GridSearchCV` (где применимо)
- Все метрики (`AUC`, `AP`, `Precision`, `Recall`, `F1`, `MCC (Matthews Correlation Coefficient)`)

## 📊 Logistic Regression (11 экспериментов)

| № | Эксперимент | AUC | Ссылка |
|---|-------------|-----|--------|
| 1 | logreg_best_model_lbfgs_l2 | **0.861843** | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/cd326b00d1b74dd18fad4c7d904ee974/output/execution) |
| 2 | logreg_cv_liblinear_l2 | 0.862310| [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/4b47451ff78b4b99b827ece2bfea03c4/output/execution) |
| 3 | logreg_cv_saga_l2 | 0.861719 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/31f5d7c550a34f1f87a516f2dfa7de5b/output/execution) |
| 4 | logreg_cv_newton-cholesky | 0.862310 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/b99d5ad769c948edac8edb3a43d01978/output/execution) |
| 5 | logreg_cv_saga_elasticnet | 0.861996 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/4f48eeea5e074e4c83ba43018931143c/output/execution) |
| 6 | logreg_cv_sag_l2 | 0.862234 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/bb51d9877cab47d2b22d35fa5942a0d5/output/execution) |
| 7 | logreg_cv_lbfgs_l2 | 0.861673 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/4c6c6a649cc1411c9555c5f31f00a2e8/output/execution) |
| 8 | logreg_cv_liblinear_l1 | 0.862311 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/e1923aa0e8d1439ea60b669e8b5a9d48/output/execution) |
| 9 | logreg_liblinear_l1_c_550 | 0.862300 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/af236856e34b46229f4737285aca4011/output/execution) |
| 10 | logreg_lbfgs_l2_c_1 | 0.857941 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/4bbb7f638ec24b4fa8b6796a7e67642e/output/execution) |
| 11 | logreg_cv_newton-cg_l2 | 0.861606 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/2f30ee71982b484e89e915a054b7c1d7/output/execution) |


## 📊 Decision Tree (2 эксперимента)

| № | Эксперимент | AUC | Ссылка |
|---|-------------|-----|--------|
| 1 | dtc_best_model | **0.854574** | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/976f001ec7d64b8db0938f39736c7956/output/execution) |
| 2 | dtc_cv | 0.854574 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/4b7d8cfeb6f7431a92d9255ed33749b9/output/execution) |

## 📊 Random Forest (3 эксперимента)

| № | Эксперимент | AUC | Ссылка |
|---|-------------|-----|--------|
| 1 | rfc_best_model | **0.864773** | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/c7e4ca2c783649b4926ebe7f47738372/output/execution) |
| 2 | rfc_cv | 0.865604 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/d4a97dabbcc84fa4982b7b41720b03a8/output/execution) |
| 3 | rfc_cv | 0.864773 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/d9de2b7253614bd7ad725b944bf63fe7/output/execution) |


## 📊 LightGBM (5 экспериментов)

| № | Эксперимент | AUC | Ссылка |
|---|-------------|-----|--------|
| 1 | lgbm_best_model | **0.868188** | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/0deaedab37614f5794a5d321051d6134/output/execution) |
| 2 | lgbm_cv | 0.868188 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/b56f74cbc7c244e98ee33855b8c95076/output/execution) |
| 3 | lgbm_cv | 0.868188 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/a77c28c4f9a44df7bfd62f9788221ff1/output/execution) |
| 4 | lgbm_cv | 0.868188 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/77f6e5c51a564a44b6a024643fc87585/output/execution) |
| 5 | lgbm_cv | 0.868188 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/a86da9a632194ca68f9a0c410c51f07e/output/execution) |

## 📊 CatBoost (2 эксперимента)

| № | Эксперимент | AUC | Ссылка |
|---|-------------|-----|--------|
| 1 | catboost_best_model | 0.868110 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/3b6e28e4dc594f5185c02cb2b9c12110/output/execution) |
| 2 | catboost_cv | 0.868110 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/c5b97df057c744b8823ebca07058af92/output/execution) |


## 📊 XGBoost (4 эксперимента)

| № | Эксперимент | AUC | Ссылка |
|---|-------------|-----|--------|
| 1 | xgb_best_model | **0.867501** | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/88e5df6dc9414d8cbf2f8046969e36e8/output/execution) |
| 2 | xgb_cv | 0.867501 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/a99d71b8f87e4b2fa830bfa1d5f0cd4c/output/execution) |
| 3 | xgb_cv | 0.867501 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/f70d8639c900459f86e0d71713117f4a/output/execution) |
| 4 | xgb_cv | 0.867501 | [🔗 ClearML](https://app.clear.ml/projects/6d09fa9bc1dd4a8a90bfba5a6582051c/experiments/3a8aa56f975048ce851507bf7b7f0f08/output/execution) |


## 📈 Что логировалось в каждом эксперименте

Каждый эксперимент в ClearML содержит:
- **Scalars:** `AUC`, `AP`, `Precision`, `Recall`, `F1`, `MCC (Matthews Correlation Coefficient)`
- **Plots:** `ROC curve`, `PR curve`, `confusion matrix`, `distribution plot`
- **Feature importance** (для tree-based)
- **Coefficients** (для линейных моделей)
- **GridSearchCV results** (где применимо)
- **Hyperparameters** (полный конфиг эксперимента)
- **Artifacts:** модель, препроцессор, скейлер, submission.csv

## 📚 Подробнее

- [Обучение моделей](../../src/README.md)
- [Конфиги Hydra](../README.md)
- [Streamlit приложение](../../app/README.md)
