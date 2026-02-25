import pandas as pd
from sklearn.model_selection import train_test_split
from utils.credit_preprocessor import CreditDataPreprocessor, CreditScaler


def load_and_preprocess(cfg):
    """
    Загружает данные и применяет препроцессинг
    """
    # Загрузка данных
    df_train = pd.read_csv(cfg.data.train_path, index_col=0)
    X_predict = pd.read_csv(cfg.data.predict_path, index_col=0)

    # Удаляем таргет из predict если есть
    if cfg.data.target_col in X_predict.columns:
        print(f"Удаляем целевую колонку '{cfg.data.target_col}' из predict данных")
        X_predict = X_predict.drop(columns=[cfg.data.target_col])

    # Отделяем таргет
    y = df_train[cfg.data.target_col]
    X = df_train.drop(columns=[cfg.data.target_col])

    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        stratify=y if cfg.training.stratify else None
    )

    # Препроцессинг
    preprocessor = CreditDataPreprocessor(**cfg.preprocessor)

    # Чистим трейн
    if cfg.cleaning.remove_outliers:
        X_train_clean, y_train_clean = preprocessor.clean_train(X_train, y_train)
    else:
        X_train_clean, y_train_clean = X_train, y_train

    # Учим и трансформируем
    preprocessor.fit(X_train_clean)

    X_train_transformed = preprocessor.transform(X_train_clean)
    X_test_transformed = preprocessor.transform(X_test)
    X_predict_transformed = preprocessor.transform(X_predict)

    # Масштабирование
    scaler = CreditScaler(scaler_type=cfg.scaler.type)
    scaler.fit(X_train_transformed)

    X_train_scaled = scaler.transform(X_train_transformed)
    X_test_scaled = scaler.transform(X_test_transformed)
    X_predict_scaled = scaler.transform(X_predict_transformed)

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'X_predict': X_predict_scaled,
        'y_train': y_train_clean,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'scaler': scaler
    }