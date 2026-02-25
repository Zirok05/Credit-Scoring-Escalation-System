import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import joblib
import os
import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from clearml import Task
from hydra.core.hydra_config import HydraConfig


from utils.data_loader import load_and_preprocess
from utils.metrics import (
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix,
    plot_distribution, plot_feature_importance,
    plot_GridSearchCV
)
Task.set_offline(True)
@hydra.main(version_base=None, config_path="../configs", config_name="experiment/lgbm_experiment")
def main(cfg: DictConfig):
    # Разворачиваем experiment в корень
    if 'experiment' in cfg:
        cfg = cfg.experiment

    # ClearML
    task_name = cfg.logging.name
    if cfg.model.use_cv:
        task_name = cfg.logging.name + '_cv'
    task = Task.init(
        project_name=cfg.logging.project,
        task_name=task_name,
        tags=cfg.logging.tags
    )
    task.connect(OmegaConf.to_container(cfg))

    # Загрузка и препроцессинг
    data = load_and_preprocess(cfg)
    print(data['X_train'].columns)

    # Обучение
    print("Обучение...")

    pos = sum(data['y_train'] == 1)
    neg = sum(data['y_train'] == 0)
    scale_pos_weight = neg/pos
    cfg.model.params.scale_pos_weight = scale_pos_weight

    model = LGBMClassifier(**cfg.model.params)

    if cfg.model.use_cv:
        cv_kwargs = OmegaConf.to_container(cfg.model.cv_params, resolve=True)
        cv_kwargs['estimator'] = model

        param_grid = cv_kwargs['param_grid']
        print(param_grid)
        n_combs = 1
        for v in param_grid.values():
            n_combs *= len(v)

        if isinstance(cv_kwargs['cv'], int):
            n_splits = cv_kwargs['cv']
            total_fits = n_combs * n_splits
            print(f"Комбинаций параметров: {n_combs}")
            print(f"CV фолдов: {n_splits}")
            print(f"Всего обучений: {total_fits}")

        grid_cv = GridSearchCV(**cv_kwargs)
        with joblib.parallel_backend('loky', verbose=5):
            grid_cv.fit(data['X_train'], data['y_train'])
        model = grid_cv.best_estimator_
        cv_results = plot_GridSearchCV(grid_cv, task, model_name=cfg.logging.name)

    else:
        time_start = time.time()
        model.fit(data['X_train'], data['y_train'])
        time_elapsed = time.time() - time_start
        print(f"fit time: {time_elapsed}")

    time_start = time.time()
    train_pred = model.predict_proba(data['X_train'])[:, 1]
    time_elapsed = time.time() - time_start
    print(f"predict time: {time_elapsed}")

    test_pred = model.predict_proba(data['X_test'])[:, 1]
    train_auc = roc_auc_score(data['y_train'], train_pred)
    test_auc = roc_auc_score(data['y_test'], test_pred)

    print(f"Train AUC: {train_auc:.6f}")
    print(f"Test AUC: {test_auc:.6f}")

    # Логирование метрик
    task.get_logger().report_scalar("AUC", "train", train_auc, 0)
    task.get_logger().report_scalar("AUC", "test", test_auc, 0)

    # Порог одобрения из конфига
    approval_threshold = cfg.approval_threshold.default

    # Графики
    plot_roc_curve(data['y_test'], test_pred, test_auc, task)
    ap = plot_pr_curve(data['y_test'], test_pred, task)
    plot_confusion_matrix(data['y_test'], test_pred, task, threshold=approval_threshold)
    plot_distribution(data['y_test'], test_pred, task)

    # Логируем AP
    task.get_logger().report_scalar("Metrics", "AP (default)", ap, 0)

    # Feature importance
    plot_feature_importance(model, data['X_train'].columns, task)

    # Сохранение
    output_dir = HydraConfig.get().runtime.output_dir
    joblib.dump(data['preprocessor'], os.path.join(output_dir, "credit_transformer.pkl"))
    joblib.dump(data['scaler'], os.path.join(output_dir, "credit_scaler.pkl"))
    joblib.dump(model, os.path.join(output_dir, f"{cfg.logging.name}_model.pkl"))

    task.upload_artifact("preprocessor", os.path.join(output_dir, "credit_transformer.pkl"))
    task.upload_artifact("scaler", os.path.join(output_dir, "credit_scaler.pkl"))
    task.upload_artifact("model", os.path.join(output_dir, f"{cfg.logging.name}_model.pkl"))

    if cfg.model.use_cv:
        cv_results.to_csv(os.path.join(output_dir, "cv_results.csv"))
        task.upload_artifact("cv_results", os.path.join(output_dir, "cv_results.csv"))

    # Сабмит для Kaggle
    predict_proba = model.predict_proba(data['X_predict'])[:, 1]
    submission = pd.DataFrame({'Id': data['X_predict'].index, 'Probability': predict_proba})
    submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
    task.upload_artifact("submission", os.path.join(output_dir, "submission.csv"))

    print(f"Готово! Сабмит сохранен в {output_dir}/submission.csv")


if __name__ == "__main__":
    main()

# python src/train_lgbm.py experiment.model.use_cv=true