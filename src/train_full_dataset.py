import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from utils.credit_preprocessor import CreditDataPreprocessor, CreditScaler

# Импорты моделей
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier

SRC_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.path.dirname(SRC_PATH)
BEST_MODELS_PATH = os.path.join(PROJECT_PATH, 'models/best/')

df_train = pd.read_csv(os.path.join(PROJECT_PATH, 'datasets/cs-training.csv'), index_col=0)
df_predict = pd.read_csv(os.path.join(PROJECT_PATH, 'datasets/cs-test.csv'), index_col=0)
X = df_train.drop('SeriousDlqin2yrs', axis=1)
df_predict = df_predict.drop('SeriousDlqin2yrs', axis=1)
y = df_train['SeriousDlqin2yrs']

preprocessor = CreditDataPreprocessor(drop_special_codes=False)
X_train_clean, y_train_clean = preprocessor.clean_train(X, y)
preprocessor.fit(X_train_clean)
X_train_transformed = preprocessor.transform(X_train_clean)
X_predict_transformed = preprocessor.transform(df_predict)

scaler = CreditScaler()
scaler.fit(X_train_transformed)
X_train_scaled = scaler.transform(X_train_transformed)
X_predict_scaled = scaler.transform(X_predict_transformed)

joblib.dump(preprocessor, os.path.join(PROJECT_PATH, 'preprocessors/preprocessor_150.pkl'))
joblib.dump(scaler, os.path.join(PROJECT_PATH, 'preprocessors/scaler_150.pkl'))
print("✅ Препроцессоры сохранены")
print("\nЗАГРУЗКА ЛУЧШИХ МОДЕЛЕЙ И ОБУЧЕНИЕ С НУЛЯ")

# Соответствие ключей и классов моделей
model_classes = {
    'logreg': LogisticRegression,
    'rfc': RandomForestClassifier,
    'xgb': xgb.XGBClassifier,
    'lgbm': lgb.LGBMClassifier,
    'catboost': CatBoostClassifier,
    'dtc': DecisionTreeClassifier
}

models_to_train = {
    'logreg': 'Logistic Regression',
    'rfc': 'Random Forest',
    'xgb': 'XGBoost',
    'lgbm': 'LightGBM',
    'catboost': 'CatBoost',
    'dtc': 'Decision Tree Classifier'
}

for model_key, model_name in models_to_train.items():
    print(f"\nОбработка {model_name}...")

    # Загружаем старую модель
    model_path = os.path.join(BEST_MODELS_PATH, f'train_120/{model_key}_model.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    old_model = joblib.load(model_path)
    print(f"Модель загружена из {model_path}")

    # Извлекаем только гиперпараметры (без атрибутов обучения)
    if hasattr(old_model, 'get_params'):
        all_params = old_model.get_params()
        # Убираем все атрибуты, которые заканчиваются на '_' (появляются после fit)
        params = {k: v for k, v in all_params.items() if not k.endswith('_')}
    else:
        params = {}
    print(model_key, params)
    # Создаем новую модель с теми же гиперпараметрами
    model = model_classes[model_key](**params)

    # Обучаем с нуля
    model.fit(X_train_scaled, y_train_clean)

    # Оценка
    y_pred = model.predict_proba(X_train_scaled)[:, 1]
    auc = roc_auc_score(y_train_clean, y_pred)
    print(f"Train AUC: {auc:.6f}")

    # Сохраняем
    output_file = os.path.join(BEST_MODELS_PATH, f'train_150/{model_key}_150_model.pkl')
    joblib.dump(model, output_file)
    print(f"Сохранено: {output_file}")

    # Сабмит для Kaggle
    predict_proba = model.predict_proba(X_predict_scaled)[:, 1]
    submission = pd.DataFrame({'Id': X_predict_scaled.index, 'Probability': predict_proba})
    output_dir = os.path.join(PROJECT_PATH, 'kaggle_submissions/best_models/')
    submission.to_csv(os.path.join(output_dir, f"{model_key}_submission.csv"), index=False)