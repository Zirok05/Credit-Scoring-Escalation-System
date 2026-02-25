import streamlit as st
import joblib
import os


@st.cache_resource
def load_artifacts(models_path, preprocessor_path):
    """Загрузка препроцессоров и моделей"""
    preprocessor = joblib.load(os.path.join(preprocessor_path, 'preprocessor_150.pkl'))
    scaler = joblib.load(os.path.join(preprocessor_path, 'scaler_150.pkl'))

    models = {}
    model_files = {
        'Logistic Regression': 'logreg_150_model.pkl',
        'XGBoost': 'xgb_150_model.pkl',
        'LightGBM': 'lgbm_150_model.pkl',
        'CatBoost': 'catboost_150_model.pkl',
        'Random Forest': 'rfc_150_model.pkl'
    }

    for name, filename in model_files.items():
        path = os.path.join(models_path, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    return preprocessor, scaler, models