import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import joblib
import os
import time
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from clearml import Task
from hydra.core.hydra_config import HydraConfig

# Импорты из utils
from utils.data_loader import load_and_preprocess
from utils.metrics import (
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix,
    plot_distribution, plot_coefficients, plot_LogisticRegressionCV,
    plot_log_metrics)

Task.set_offline(True)
@hydra.main(version_base=None, config_path="../configs", config_name="experiment/logreg_experiment")
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

    # Обучение
    print("Обучение...")
    if cfg.model.use_cv:

        print(cfg.model.cv_params)

        cv_kwargs = OmegaConf.to_container(cfg.model.cv_params, resolve=True)
        model = LogisticRegressionCV(**cv_kwargs)

    else:
        model = LogisticRegression(**cfg.model.params)

    time_start = time.time()
    model.fit(data['X_train'], data['y_train'])
    time_elapsed = time.time() - time_start
    print(f"fit time: {time_elapsed}")

    if cfg.model.use_cv:
        plot_LogisticRegressionCV(model, task)

    # Оценка
    time_start = time.time()
    train_pred = model.predict_proba(data['X_train'])[:, 1]
    time_elapsed = time.time() - time_start
    print(f"predict time: {time_elapsed}")
    test_pred = model.predict_proba(data['X_test'])[:, 1]
    train_auc = roc_auc_score(data['y_train'], train_pred)
    test_auc = roc_auc_score(data['y_test'], test_pred)

    print(f"Train AUC: {train_auc:.6f}")
    print(f"Test AUC: {test_auc:.6f}")
    print(model.get_params())
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

    plot_log_metrics(data['y_test'], test_pred, task, dataset_name="test", threshold=approval_threshold)

    # Коэффициенты для линейной модели
    plot_coefficients(model, data['X_train'].columns, task)

    # Сохранение
    output_dir = HydraConfig.get().runtime.output_dir
    joblib.dump(data['preprocessor'], os.path.join(output_dir, "credit_transformer.pkl"))
    joblib.dump(data['scaler'], os.path.join(output_dir, "credit_scaler.pkl"))
    joblib.dump(model, os.path.join(output_dir, f"{cfg.logging.name}_model.pkl"))

    task.upload_artifact("preprocessor", os.path.join(output_dir, "credit_transformer.pkl"))
    task.upload_artifact("scaler", os.path.join(output_dir, "credit_scaler.pkl"))
    task.upload_artifact("model", os.path.join(output_dir, f"{cfg.logging.name}_model.pkl"))

    # Сабмит для Kaggle
    predict_proba = model.predict_proba(data['X_predict'])[:, 1]
    submission = pd.DataFrame({'Id': data['X_predict'].index, 'Probability': predict_proba})
    submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
    task.upload_artifact("submission", os.path.join(output_dir, "submission.csv"))

    print(f"Готово! Сабмит сохранен в {output_dir}/submission.csv")


if __name__ == "__main__":
    main()

# python src/train_logreg.py experiment.model.use_cv=true