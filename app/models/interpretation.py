import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_DESCRIPTIONS = { ... }

def get_feature_display_name(feature_name):
    if feature_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_name]
    name = feature_name.replace('_', ' ').title()
    name = name.replace('Over', '>')
    name = name.replace('Loans', 'Кредитов')
    return name


def interpret_lr(features, lr_model, feature_names):
    """Интерпретация логистической регрессии"""
    if isinstance(features, np.ndarray):
        features = pd.DataFrame(features, columns=feature_names)
    coefficients = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'value': features.iloc[0].values
    })
    importance_df['logit_contribution'] = importance_df['coefficient'] * importance_df['value']
    importance_df['abs_logit'] = abs(importance_df['logit_contribution'])
    importance_df = importance_df.sort_values('abs_logit', ascending=False)

    base_proba = lr_model.predict_proba(features)[0, 1]
    marginal_effects = []
    features_array = features.values

    for i, feature in enumerate(feature_names):
        features_zero = features_array.copy()
        features_zero[0, i] = 0
        zero_proba = lr_model.predict_proba(features_zero)[0, 1]
        marginal_effect = base_proba - zero_proba
        marginal_effects.append({
            'feature': feature,
            'marginal_effect': marginal_effect,
            'abs_marginal': abs(marginal_effect)
        })

    marginal_df = pd.DataFrame(marginal_effects).sort_values('abs_marginal', ascending=False)

    logit = intercept + importance_df['logit_contribution'].sum()
    proba = 1 / (1 + np.exp(-logit))

    return {
        'logit_contributions': importance_df,
        'marginal_effects': marginal_df,
        'probability': proba,
        'logit': logit,
        'intercept': intercept
    }

def plot_feature_importance_sns(importance_df, value_col='logit_contribution', title="Вклад признаков в логит"):
    df = importance_df.head(10).copy()
    df = df.sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    colors = ['#d7191c' if x > 0 else '#1a9641' if x < 0 else '#ffffbf' for x in df[value_col]]
    bars = ax.barh(df['feature'], df[value_col], color=colors, edgecolor='white', linewidth=1.5, alpha=0.9)

    for bar, val in zip(bars, df[value_col]):
        if abs(val) > 0.02:
            x_pos = val - 0.02 if val > 0 else val + 0.02
            ha = 'right' if val > 0 else 'left'
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', ha=ha, va='center', fontsize=9)

    ax.axvline(x=0, color='#495057', linestyle='-', linewidth=1, alpha=0.3)
    ax.grid(axis='x', alpha=0.15, linestyle='--', color='#adb5bd')
    ax.set_axisbelow(True)
    ax.set_xlabel('Вклад в логит', fontsize=11)
    ax.set_ylabel('')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.set_yticklabels([get_feature_display_name(x) for x in df['feature']], fontsize=10)
    ax.set_yticklabels([get_feature_display_name(x) for x in df['feature']], fontsize=10)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.tight_layout()
    return fig

def plot_marginal_effects_sns(marginal_df, title="Влияние на вероятность дефолта"):
    df = marginal_df.head(10).copy()
    df = df.sort_values('marginal_effect', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    colors = ['#d7191c' if x > 0 else '#1a9641' if x < 0 else '#ffffbf' for x in df['marginal_effect']]
    bars = ax.barh(df['feature'], df['marginal_effect'], color=colors, edgecolor='white', linewidth=1.5, alpha=0.9)

    for bar, val in zip(bars, df['marginal_effect']):
        if abs(val) > 0.01:
            x_pos = val - 0.01 if val > 0 else val + 0.01
            ha = 'right' if val > 0 else 'left'
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{val:.1%}', ha=ha, va='center', fontsize=9)

    ax.axvline(x=0, color='#495057', linestyle='-', linewidth=1, alpha=0.3)
    ax.grid(axis='x', alpha=0.15, linestyle='--', color='#adb5bd')
    ax.set_axisbelow(True)
    ax.set_xlabel('Изменение вероятности', fontsize=11)
    ax.set_ylabel('')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax.set_yticklabels([get_feature_display_name(x) for x in df['feature']], fontsize=10)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.tight_layout()
    return fig


def plot_shap_analysis(second_model, processed_scaled, feature_names, second_model_name):
    """Отображение SHAP анализа для tree-based моделей"""
    import streamlit as st
    st.markdown("---")
    st.subheader(f"⚡ Детальный анализ: {second_model_name} (SHAP)")

    with st.spinner("🔄 Рассчитываем SHAP значения..."):
        try:
            import shap

            # Создаем explainer и считаем SHAP
            explainer = shap.TreeExplainer(second_model)
            shap_values = explainer.shap_values(processed_scaled)

            # Для бинарной классификации
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # 1. Waterfall plot
            fig, ax = plt.subplots(figsize=(12, 7))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=processed_scaled.iloc[0].values,
                    feature_names=feature_names
                ),
                show=False,
            )
            plt.tight_layout()
            st.pyplot(fig)

            # 2. Объяснение как читать график
            with st.expander("📋 Как читать SHAP график?"):
                st.markdown("""
                - **f(x)** = итоговое предсказание модели
                - **base value** = среднее предсказание по всем клиентам
                - 🔴 Красное → признаки, повышающие риск
                - 🔵 Синее → признаки, снижающие риск
                """)

            # 3. Таблица с SHAP значениями
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'shap_value': shap_values[0],
                'abs_shap': abs(shap_values[0])
            }).sort_values('abs_shap', ascending=False)

            shap_df['description'] = shap_df['feature'].apply(get_feature_display_name)

            st.markdown("### 📋 Факторы, влияющие на решение:")

            col1, col2 = st.columns(2)

            with col1:
                pos = shap_df[shap_df['shap_value'] > 0].head(5)
                if len(pos) > 0:
                    st.markdown("**🔴 Повышают риск:**")
                    for _, row in pos.iterrows():
                        st.markdown(f"- {row['description']}: +{row['shap_value']:.3f}")

            with col2:
                neg = shap_df[shap_df['shap_value'] < 0].head(5)
                if len(neg) > 0:
                    st.markdown("**🟢 Снижают риск:**")
                    for _, row in neg.iterrows():
                        st.markdown(f"- {row['description']}: {row['shap_value']:.3f}")

            with st.expander("📋 Все SHAP значения"):
                display_df = shap_df[['feature', 'description', 'shap_value']].copy()
                display_df.columns = ['Признак', 'Описание', 'SHAP']
                display_df['SHAP'] = display_df['SHAP'].round(3)
                st.dataframe(display_df.sort_values('SHAP', ascending=False), width='stretch')

        except Exception as e:
            st.error(f"❌ Ошибка SHAP: {e}")
            st.info("Установите shap: `pip install shap`")
