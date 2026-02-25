import numpy as np
def check_business_rules(df):
    """
    Батчевая проверка бизнес-правил

    Возвращает:
    - manual_mask: булев массив (True = в ручной разбор)
    - auto_reject_mask: булев массив (True = сразу отказ)
    - messages: массив сообщений
    - auto_decisions: массив решений для auto_reject_mask (всегда 1 - отказ)
    """
    n = len(df)
    manual_mask = np.zeros(n, dtype=bool)
    auto_reject_mask = np.zeros(n, dtype=bool)
    messages = [''] * n
    auto_decisions = np.zeros(n, dtype=int)

    # Извлекаем колонки
    age = df['age'].fillna(0).values
    monthly_income = df['MonthlyIncome'].fillna(0).values
    debt_ratio = df['DebtRatio'].fillna(0).values
    monthly_debt = np.where(monthly_income > 0,
                            debt_ratio * monthly_income,
                            debt_ratio)

    late_90 = df['NumberOfTimes90DaysLate'].fillna(0).values
    late_60_89 = df['NumberOfTime60-89DaysPastDueNotWorse'].fillna(0).values
    late_30_59 = df['NumberOfTime30-59DaysPastDueNotWorse'].fillna(0).values

    real_estate = df['NumberRealEstateLoansOrLines'].fillna(0).values
    utilization = df['RevolvingUtilizationOfUnsecuredLines'].fillna(0).values

    # 1. КРИТИЧЕСКИЕ ПРАВИЛА - сразу отказ
    mask = (age < 18)
    auto_reject_mask[mask] = True
    auto_decisions[mask] = 1
    messages = np.where(mask, 'Возраст менее 18 лет - кредит не выдаётся', messages)

    # 2. СПЕЦИАЛЬНЫЕ БАНКОВСКИЕ КОДЫ - сразу ручной разбор
    mask = (late_90 == 98) | (late_60_89 == 98) | (late_30_59 == 98)
    manual_mask[mask] = True
    messages = np.where(mask, 'Код 98: Списание долга как безнадежного', messages)

    mask = (late_90 == 96) | (late_60_89 == 96) | (late_30_59 == 96)
    manual_mask[mask] = True
    messages = np.where(mask, 'Код 96: Изъятие залога или реализация имущества', messages)

    # 3. КРИТИЧЕСКИЕ ПРАВИЛА - сразу ручной разбор
    mask = (age > 80)
    manual_mask[mask] = True
    messages = np.where(mask, 'Возраст > 80 лет - требуется ручной разбор (индивидуальные условия)', messages)

    mask = (monthly_income > 1000000)
    manual_mask[mask] = True
    messages = np.where(mask, 'Доход свыше 1,000,000 $ - требуется ручной разбор', messages)

    mask = (monthly_debt > 1000000)
    manual_mask[mask] = True
    messages = np.where(mask, 'Платежи свыше 1,000,000 $ - требуется ручной разбор', messages)

    mask = (utilization > 2)
    manual_mask[mask] = True
    messages = np.where(mask, 'Использование кредитных средств превышает 200%', messages)

    mask = (real_estate > 20)
    manual_mask[mask] = True
    messages = np.where(mask, 'Количество кредитов под залог недвижимости слишком велико - ручной разбор', messages)

    return manual_mask, auto_reject_mask, messages, auto_decisions


def escalation_decision(applications_df, lr_model, second_model, second_model_name,
                        threshold=0.5, lr_margins=[0.35], second_margins=[0.4],
                        preprocessor=None, scaler=None):
    """
    Универсальная эскалационная логика

    1. Бизнес-правила:
       - часть заявок сразу в ручной разбор
       - часть заявок сразу отказ
    2. Оставшиеся -> LR
    3. Если LR неуверена -> вторая модель
    """
    n = len(applications_df)
    decisions = [None] * n
    manual_mask = np.zeros(n, dtype=bool)

    # СЧЁТЧИКИ
    stats = {
        'business_manual': 0,  # ручной разбор по бизнес-правилам
        'business_auto': 0,  # авто отказ по бизнес-правилам
        'lr_confident': 0,  # уверенно решены LR
        'second_confident': 0,  # уверенно решены второй моделью
        'second_uncertain': 0,  # неуверенность второй модели
        'total': n
    }

    # 1. Бизнес-правила
    bus_manual_mask, bus_reject_mask, bus_messages, bus_decisions = check_business_rules(applications_df)

    # Обрабатываем сразу отказ
    for i in range(n):
        if bus_reject_mask[i]:
            stats['business_auto'] += 1
            decisions[i] = {
                'final_decision': 1,
                'model_used': 'Business Rules',
                'needs_review': False,
                'probability': 1.0,
                'message': bus_messages[i],
                'lr_proba': None,
                'second_proba': None,
                'decision_path': [f"❌ Бизнес-правила: {bus_messages[i]}"]
            }

    # Обрабатываем сразу ручной разбор
    for i in range(n):
        if bus_manual_mask[i]:
            stats['business_manual'] += 1
            manual_mask[i] = True
            decisions[i] = {
                'final_decision': None,
                'model_used': 'Business Rules',
                'needs_review': True,
                'probability': None,
                'message': bus_messages[i],
                'lr_proba': None,
                'second_proba': None,
                'decision_path': [f"⚠️ Бизнес-правила: {bus_messages[i]}"]
            }

    # 2. Заявки, которые идут к моделям (не отсеялись бизнес-правилами)
    model_indices = [i for i in range(n) if decisions[i] is None]

    if not model_indices:
        return decisions, manual_mask, stats

    # 3. Обработка моделями
    df_models = applications_df.iloc[model_indices]

    # Препроцессинг
    processed = preprocessor.transform(df_models)
    processed_scaled = scaler.transform(processed)

    # LR предсказания (батч)
    lr_probas = lr_model.predict_proba(processed_scaled)[:, 1]

    # Определяем отступы для LR
    if len(lr_margins) == 1:
        lr_low = lr_high = lr_margins[0]
    else:
        lr_low, lr_high = lr_margins[0], lr_margins[1]

    # Проверяем уверенность LR
    lr_confident = np.zeros(len(model_indices), dtype=bool)
    lr_margin_values = np.zeros(len(model_indices))

    for j, proba in enumerate(lr_probas):
        if proba < threshold:
            margin = threshold - proba
            lr_confident[j] = margin >= lr_low
        else:
            margin = proba - threshold
            lr_confident[j] = margin >= lr_high
        lr_margin_values[j] = margin

    # Обрабатываем уверенные LR
    for j, idx in enumerate(model_indices):
        if lr_confident[j]:
            stats['lr_confident'] += 1
            decisions[idx] = {
                'final_decision': int(lr_probas[j] >= threshold),
                'probability': lr_probas[j],
                'model_used': 'Logistic Regression',
                'needs_review': False,
                'lr_proba': lr_probas[j],
                'second_proba': None,
                'lr_margin': lr_margin_values[j],
                'lr_confident': True,
                'second_used': False,
                'decision_path': [
                    f"1️⃣ Logistic Regression: {lr_probas[j]:.1%} (отступ: {lr_margin_values[j]:.1%})",
                    f"   ✅ LR уверена - финальное решение"
                ]
            }

    # Неуверенные LR - идут ко второй модели
    uncertain_indices = [model_indices[j] for j in range(len(model_indices)) if not lr_confident[j]]

    if uncertain_indices:
        # Находим позиции неуверенных заявок
        uncertain_positions = [j for j in range(len(model_indices)) if not lr_confident[j]]
        processed_uncertain_scaled = processed_scaled.iloc[uncertain_positions]

        # Вторая модель (батч)
        second_probas = second_model.predict_proba(processed_uncertain_scaled)[:, 1]

        # Определяем отступы для второй модели
        if len(second_margins) == 1:
            second_low = second_high = second_margins[0]
        else:
            second_low, second_high = second_margins[0], second_margins[1]

        # Проверяем уверенность второй модели
        for k, idx in enumerate(uncertain_indices):
            proba = second_probas[k]
            if proba < threshold:
                second_margin = threshold - proba
                second_confident = second_margin >= second_low
            else:
                second_margin = proba - threshold
                second_confident = second_margin >= second_high

            # Формируем decision_path
            path = [
                f"1️⃣ Logistic Regression: {lr_probas[uncertain_positions[k]]:.1%} (отступ: {lr_margin_values[uncertain_positions[k]]:.1%})",
                f"   ⚠️ LR не уверена → вызываем {second_model_name}",
                f"2️⃣ {second_model_name}: {proba:.1%} (отступ: {second_margin:.1%})"
            ]

            if second_confident:
                stats['second_confident'] += 1
                path.append(f"   ✅ {second_model_name} уверен - финальное решение")
                decisions[idx] = {
                    'final_decision': int(proba >= threshold),
                    'probability': proba,
                    'model_used': second_model_name,
                    'needs_review': False,
                    'lr_proba': lr_probas[uncertain_positions[k]],
                    'second_proba': proba,
                    'lr_margin': lr_margin_values[uncertain_positions[k]],
                    'second_margin': second_margin,
                    'lr_confident': False,
                    'second_confident': True,
                    'second_used': True,
                    'decision_path': path
                }
            else:
                stats['second_uncertain'] += 1
                path.append(f"   ⚠️ {second_model_name} не уверен → ручной разбор")
                manual_mask[idx] = True
                decisions[idx] = {
                    'final_decision': None,
                    'probability': proba,
                    'model_used': 'Manual Review',
                    'needs_review': True,
                    'lr_proba': lr_probas[uncertain_positions[k]],
                    'second_proba': proba,
                    'lr_margin': lr_margin_values[uncertain_positions[k]],
                    'second_margin': second_margin,
                    'lr_confident': False,
                    'second_confident': False,
                    'second_used': True,
                    'message': 'Модели не уверены в решении',
                    'decision_path': path
                }

    return decisions, manual_mask, stats