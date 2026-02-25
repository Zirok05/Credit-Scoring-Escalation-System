import streamlit as st
import pandas as pd
import os
from app.utils.data_loader import load_artifacts
from app.models.escalation import escalation_decision
from app.models.interpretation import (
    interpret_lr, plot_feature_importance_sns,
    plot_marginal_effects_sns, plot_shap_analysis,
    get_feature_display_name
)

# Пути
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_PATH = os.path.join(PROJECT_PATH, 'models/best/train_150/')
PREPROCESSOR_PATH = os.path.join(PROJECT_PATH, 'preprocessors/')


def main():
    st.title("🏦 Кредитный скоринг - Анкета")

    # Загрузка артефактов
    preprocessor, scaler, models = load_artifacts(MODELS_PATH, PREPROCESSOR_PATH)

    # Инициализация статистики
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'total': 0,
            'manual': 0,
            'lr_confident': 0,
            'second_used': 0,
            'second_confident': 0,
            'approved': 0,
            'declined': 0
        }

    if 'step' not in st.session_state:
        st.session_state.step = 'input'

    # ВВОД ДАННЫХ
    if st.session_state.step == 'input':
        st.header("📋 Анкета заемщика")

        with st.form("credit_form"):
            st.subheader("👤 Личная информация")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Возраст", 0, 150, 35)
            with col2:
                dependents = st.number_input("Иждивенцы", 0, 20, 0)

            st.subheader("💰 Ежемесячный доход")
            income_method = st.radio("Способ указания дохода", ["Слайдер (до 20,000$)", "Точное значение"],
                                     horizontal=True)

            st.subheader("💳 Ежемесячные платежи")
            debt_method = st.radio("Способ указания платежей", ["Слайдер (до 10,000$)", "Точное значение"],
                                   horizontal=True)

            st.subheader("📊 Кредитная история")
            credit_lines = st.number_input("Открытых кредитов и карт", 0, 100, 5)
            real_estate = st.number_input("Кредитов под залог недвижимости", 0, 100, 1)

            st.subheader("📈 Использование лимитов")
            util_method = st.radio("Уровень использования",
                                   ["Норма (0-100%)", "Овердрафт (100-200%)", "Экстремальный (>200%)"], horizontal=True)

            st.subheader("⏱️ Просрочки за последние 2 года")
            col1, col2, col3 = st.columns(3)
            with col1:
                late_30_59 = st.number_input("30-59 дней", 0, 100, 0)
            with col2:
                late_60_89 = st.number_input("60-89 дней", 0, 100, 0)
            with col3:
                late_90 = st.number_input("90+ дней", 0, 100, 0)

            submitted = st.form_submit_button("➡️ Далее: указать точные значения")

        if submitted:
            st.session_state.update({
                'age': age, 'dependents': dependents, 'income_method': income_method,
                'debt_method': debt_method, 'credit_lines': credit_lines,
                'real_estate': real_estate, 'util_method': util_method,
                'late_30_59': late_30_59, 'late_60_89': late_60_89, 'late_90': late_90
            })
            st.session_state.step = 'values'
            st.rerun()


    # ВВОД ТОЧНЫХ ЗНАЧЕНИЙ
    elif st.session_state.step == 'values':
        st.header("💰 Укажите точные значения")

        with st.form("values_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Доход")
                if st.session_state.income_method == "Слайдер (до 20,000$)":
                    monthly_income = st.slider("Ежемесячный доход ($)", 0, 20000, 5000)
                else:
                    monthly_income = st.number_input("Ежемесячный доход ($)", 0, 1000000, 5000)

            with col2:
                st.subheader("Платежи")
                if st.session_state.debt_method == "Слайдер (до 10,000$)":
                    monthly_debt = st.slider("Ежемесячные платежи ($)", 0, 10000, 1500)
                else:
                    monthly_debt = st.number_input("Ежемесячные платежи ($)", 0, 1000000, 1500)

            st.subheader("📈 Использование лимитов")
            if st.session_state.util_method == "Норма (0-100%)":
                util_value = st.slider("Процент использования", 0, 100, 20)
                utilization = util_value / 100
            elif st.session_state.util_method == "Овердрафт (100-200%)":
                util_value = st.slider("Процент использования", 100, 200, 120)
                utilization = util_value / 100
            else:
                st.warning("Экстремальное использование (>200%) - автоматический ручной разбор")
                utilization = st.number_input("Процент использования", 200, 1000, 200) / 100

            submitted = st.form_submit_button("✅ Получить решение")

            # САЙДБАР
            with st.sidebar:
                st.markdown("---")
                st.subheader("⚙️ Настройки")

                with st.expander("🎯 Пороги уверенности", expanded=False):
                    threshold = st.slider("Порог одобрения", 0.3, 0.7, 0.5, 0.05)
                    lr_margin = st.slider("Отступ LR", 0.2, 0.5, 0.35, 0.05)
                    second_margin = st.slider("Отступ второй модели", 0.2, 0.5, 0.4, 0.05)

                with st.expander("🤖 Выбор модели", expanded=False):
                    available_models = [name for name in models.keys() if name != 'Logistic Regression']
                    second_model_name = st.selectbox("Модель для эскалации", available_models)

                with st.expander("📊 Статистика", expanded=False):
                    stats = st.session_state.stats
                    if stats['total'] > 0:
                        st.metric("Всего заявок", stats['total'])
                        st.metric("Ручной разбор", f"{stats['manual'] / stats['total']:.1%}")
                        st.metric("LR уверена", f"{stats['lr_confident'] / stats['total']:.1%}")
                        if stats['second_used'] > 0:
                            st.metric("Вторая модель уверена",
                                      f"{stats['second_confident'] / stats['second_used']:.1%}")

                        if st.button("🔄 Сброс"):
                            st.session_state.stats = {'total': 0, 'manual': 0, 'lr_confident': 0,
                                                      'second_used': 0, 'second_confident': 0,
                                                      'approved': 0, 'declined': 0}
                            st.rerun()
                    else:
                        st.info("Нет данных")

                with st.expander("ℹ️ О проекте", expanded=False):
                    st.markdown(f"""
                        **Модели:**
                        - Logistic Regression
                        - {', '.join(available_models)}

                        **AUC:** 0.8578 (LR), ~0.87 (остальные)
                    """)

            st.session_state.threshold = threshold
            st.session_state.lr_margin = lr_margin
            st.session_state.second_margin = second_margin
            st.session_state.second_model_name = second_model_name

        if submitted:
            debt_ratio = monthly_debt / monthly_income if monthly_income > 0 else monthly_debt

            # Подготовка данных
            input_data = pd.DataFrame([{
                'RevolvingUtilizationOfUnsecuredLines': utilization,
                'age': st.session_state.age,
                'NumberOfTime30-59DaysPastDueNotWorse': st.session_state.late_30_59,
                'DebtRatio': debt_ratio,
                'MonthlyIncome': monthly_income,
                'NumberOfOpenCreditLinesAndLoans': st.session_state.credit_lines,
                'NumberOfTimes90DaysLate': st.session_state.late_90,
                'NumberRealEstateLoansOrLines': st.session_state.real_estate,
                'NumberOfTime60-89DaysPastDueNotWorse': st.session_state.late_60_89,
                'NumberOfDependents': st.session_state.dependents
            }])

            st.markdown("---")

            with st.spinner("🔄 Анализ заявки..."):
                lr_model = models['Logistic Regression']
                second_model = models[second_model_name]

                # Единый вызов эскалации
                decisions, manual_mask, task = escalation_decision(
                    input_data,
                    lr_model,
                    second_model,
                    second_model_name,
                    threshold=st.session_state.threshold,
                    lr_margins=[st.session_state.lr_margin],
                    second_margins=[st.session_state.second_margin],
                    preprocessor=preprocessor,
                    scaler=scaler
                )
                decision = decisions[0]

                processed = preprocessor.transform(input_data)
                processed_scaled = scaler.transform(processed)

                # Обновление статистики
                st.session_state.stats['total'] += 1
                if decision['needs_review']:
                    st.session_state.stats['manual'] += 1
                else:
                    if decision['final_decision'] == 0:
                        st.session_state.stats['approved'] += 1
                    else:
                        st.session_state.stats['declined'] += 1

                if decision.get('lr_confident', False):
                    st.session_state.stats['lr_confident'] += 1

                if decision.get('second_used', False):
                    st.session_state.stats['second_used'] += 1
                    if decision.get('second_confident', False):
                        st.session_state.stats['second_confident'] += 1

                # ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
                st.subheader("🔄 Цепочка принятия решения")
                for step in decision['decision_path']:
                    st.write(step)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🏦 Logistic Regression**")
                    st.metric("Вероятность", f"{decision['lr_proba']:.1%}")
                    st.write(f"Отступ: {decision['lr_margin']:.1%}")
                    if decision['lr_confident']:
                        st.success("✅ Уверена")
                    else:
                        st.warning("⚠️ Не уверена")

                with col2:
                    st.markdown(f"**⚡ {second_model_name}**")
                    if decision['second_used']:
                        st.metric("Вероятность", f"{decision['second_proba']:.1%}")
                        st.write(f"Отступ: {decision['second_margin']:.1%}")
                        if decision['second_confident']:
                            st.success("✅ Уверен")
                        else:
                            st.warning("⚠️ Не уверен")
                    else:
                        st.info("⏳ Не вызывался")

                st.markdown("---")
                if decision['needs_review']:
                    st.warning("👨‍💼 **РУЧНОЙ РАЗБОР**")
                    st.info("Модели не уверены - требуется проверка специалистом")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        if decision['final_decision'] == 0:
                            st.success("✅ **КРЕДИТ ОДОБРЕН**")
                        else:
                            st.error("❌ **КРЕДИТ НЕ ОДОБРЕН**")
                    with col2:
                        st.metric("Модель", decision['model_used'])

                # ДЕТАЛЬНЫЙ АНАЛИЗ LR
                st.markdown("---")
                st.subheader("🔍 Детальный анализ: Logistic Regression")

                feature_names = processed_scaled.columns.tolist()
                interpretation = interpret_lr(processed_scaled, lr_model, feature_names)

                tab1, tab2 = st.tabs(["📊 Вклад в логит", "📈 Влияние на вероятность"])

                with tab1:
                    st.markdown("🔴 Положительный вклад = ↑ риск, 🟢 Отрицательный = ↓ риск")
                    fig1 = plot_feature_importance_sns(interpretation['logit_contributions'])
                    st.pyplot(fig1)

                    with st.expander("📋 Все вклады"):
                        display_df = interpretation['logit_contributions'][
                            ['feature', 'value', 'coefficient', 'logit_contribution']].copy()
                        display_df['Описание'] = display_df['feature'].apply(get_feature_display_name)
                        display_df = display_df[['Описание', 'value', 'coefficient', 'logit_contribution']]
                        display_df.columns = ['Признак', 'Значение', 'Коэф', 'Вклад']
                        display_df = display_df.round(3)
                        st.dataframe(display_df)

                with tab2:
                    st.markdown("🔴 Положительное = фактор ↑ риск, 🟢 Отрицательное = ↓ риск")
                    fig2 = plot_marginal_effects_sns(interpretation['marginal_effects'])
                    st.pyplot(fig2)

                    with st.expander("📋 Все эффекты"):
                        display_df = interpretation['marginal_effects'][['feature', 'marginal_effect']].copy()
                        display_df['Описание'] = display_df['feature'].apply(get_feature_display_name)
                        display_df = display_df[['Описание', 'marginal_effect']]
                        display_df.columns = ['Признак', 'Влияние']
                        display_df['Влияние'] = display_df['Влияние'].map('{:.1%}'.format)
                        st.dataframe(display_df)

                st.info(f"Итоговая вероятность дефолта (LR): {interpretation['probability']:.1%}")

                # ДЕТАЛЬНЫЙ АНАЛИЗ ВТОРОЙ МОДЕЛИ (SHAP для tree-based)
                if decision['second_used'] and second_model_name in ['XGBoost', 'LightGBM', 'Random Forest', 'CatBoost']:
                    plot_shap_analysis(second_model, processed_scaled, feature_names, second_model_name)

        # КНОПКА НАЗАД
        if st.button("◀️ Вернуться к выбору способов"):
            st.session_state.step = 'input'
            st.rerun()

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🏠 На главную", use_container_width=True):
            st.switch_page("main.py")

    st.markdown("---")
    st.caption("🏦 GiveMeSomeCredit - Интерпретируемый кредитный скоринг | Модели: Logistic Regression + выбор")


if __name__ == "__main__":
    main()
