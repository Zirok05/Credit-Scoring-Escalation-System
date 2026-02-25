import matplotlib.pyplot as plt
import numpy as np

def minutes_to_time(minutes, start_time="00:00"):
    """Преобразует минуты от старта в строку времени ЧЧ:ММ"""
    start_hour, start_min = map(int, start_time.split(':'))
    total_minutes = start_hour * 60 + start_min + minutes
    hour = (total_minutes // 60) % 24
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def plot_queue_dynamics(queue_history, business_queue_history=None, start_time="00:00"):
    """
    Два отдельных графика для очередей с временной шкалой ЧЧ:ММ
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Создаем метки времени для каждого часа
    total_minutes = len(queue_history)
    hours = range(0, total_minutes, 60)  # каждый час
    hour_labels = [minutes_to_time(m, start_time) for m in hours]

    # График 1: Очередь моделей
    ax1.plot(range(total_minutes), queue_history, 'b-', linewidth=1.5)
    ax1.set_xticks(hours)
    ax1.set_xticklabels(hour_labels, rotation=45)
    ax1.set_xlabel('Время')
    ax1.set_ylabel('Размер очереди')
    ax1.set_title('Очередь моделей')
    ax1.grid(True, alpha=0.3)

    # График 2: Очередь бизнес-правил
    if business_queue_history and len(business_queue_history) > 0:
        ax2.plot(range(total_minutes), business_queue_history, 'orange', linewidth=1.5)
        ax2.set_xticks(hours)
        ax2.set_xticklabels(hour_labels, rotation=45)
        ax2.set_xlabel('Время')
        ax2.set_ylabel('Размер очереди')
        ax2.set_title('Очередь бизнес-правил')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Очередь бизнес-правил')
        ax2.set_xlabel('Время')

    plt.tight_layout()
    return plt


def plot_specialist_load(specialist_busy_history, specialists_count, start_time="00:00"):
    """График загрузки специалистов с временной шкалой ЧЧ:ММ"""
    load_percent = [busy / specialists_count * 100 for busy in specialist_busy_history]

    fig, ax = plt.subplots(figsize=(10, 4))

    total_minutes = len(load_percent)
    hours = range(0, total_minutes, 60)  # каждый час
    hour_labels = [minutes_to_time(m, start_time) for m in hours]

    ax.plot(range(total_minutes), load_percent, 'g-', linewidth=1.5)
    ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Максимум')
    ax.axhline(y=80, color='b', linestyle='--', alpha=0.5, label='Цель 80%')

    ax.set_xticks(hours)
    ax.set_xticklabels(hour_labels, rotation=45)
    ax.set_xlabel('Время')
    ax.set_ylabel('Загрузка (%)')
    ax.set_title('Загрузка специалистов')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    return plt


def plot_inflow(minute_counts, start_time="00:00"):
    """
    График входящего потока заявок с заливкой под кривой
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    total_minutes = len(minute_counts)
    minutes = range(total_minutes)

    # Заливка под кривой (area plot)
    ax.fill_between(minutes, minute_counts, alpha=0.3, color='blue', label='Общий поток')

    # Основной график (линия поверх заливки)
    ax.plot(minutes, minute_counts, 'b-', linewidth=1.5, alpha=0.8)

    # Скользящее среднее
    window = 30
    if total_minutes > window:
        smoothed = np.convolve(minute_counts, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, total_minutes), smoothed,
                'r-', linewidth=2.5, label=f'Среднее за 30 мин')

    # Метки времени
    hours = range(0, total_minutes, 60)
    hour_labels = [minutes_to_time(m, start_time) for m in hours]

    ax.set_xticks(hours)
    ax.set_xticklabels(hour_labels, rotation=45)
    ax.set_xlabel('Время')
    ax.set_ylabel('Количество заявок')
    ax.set_title('Входящий поток заявок')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Горизонтальная линия среднего
    mean_value = np.mean(minute_counts)
    ax.axhline(y=mean_value, color='gray', linestyle='--', alpha=0.7,
               label=f'Среднее: {mean_value:.1f}')

    plt.tight_layout()
    return plt


def minutes_to_time(minutes, start_time="00:00"):
    """Преобразует минуты от старта в строку времени ЧЧ:ММ"""
    start_hour, start_min = map(int, start_time.split(':'))
    total_minutes = start_hour * 60 + start_min + minutes
    hour = (total_minutes // 60) % 24
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def plot_detailed_decisions(batch_stats, second_model_name="XGBoost", start_time="00:00"):
    """
    Набор графиков для каждого типа решений отдельно с временной шкалой ЧЧ:ММ
    """
    if not batch_stats:
        return None

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    times = [stat['time'] for stat in batch_stats]  # минуты
    total_minutes = max(times) if times else 0

    # Метки времени каждый час
    hours = range(0, total_minutes + 60, 60)
    hour_labels = [minutes_to_time(m, start_time) for m in hours]

    # 1. Бизнес-правила (ручной разбор)
    axes[0, 0].plot(times, [stat['business_manual'] for stat in batch_stats],
                    'r-', linewidth=1.5)
    axes[0, 0].fill_between(times, 0, [stat['business_manual'] for stat in batch_stats],
                            alpha=0.2, color='red')
    axes[0, 0].set_title('Ручной разбор: бизнес-правила', fontweight='bold')
    axes[0, 0].set_xticks(hours)
    axes[0, 0].set_xticklabels(hour_labels, rotation=45)
    axes[0, 0].set_xlabel('Время')
    axes[0, 0].set_ylabel('Заявок')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Бизнес-правила (авто отказ)
    axes[0, 1].plot(times, [stat['business_auto'] for stat in batch_stats],
                    'darkred', linewidth=1.5)
    axes[0, 1].fill_between(times, 0, [stat['business_auto'] for stat in batch_stats],
                            alpha=0.2, color='darkred')
    axes[0, 1].set_title('Авто отказ: бизнес-правила', fontweight='bold')
    axes[0, 1].set_xticks(hours)
    axes[0, 1].set_xticklabels(hour_labels, rotation=45)
    axes[0, 1].set_xlabel('Время')
    axes[0, 1].set_ylabel('Заявок')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. LR уверенные решения
    axes[1, 0].plot(times, [stat['lr_confident'] for stat in batch_stats],
                    'blue', linewidth=1.5)
    axes[1, 0].fill_between(times, 0, [stat['lr_confident'] for stat in batch_stats],
                            alpha=0.2, color='blue')
    axes[1, 0].set_title('Уверенные решения: Logistic Regression', fontweight='bold')
    axes[1, 0].set_xticks(hours)
    axes[1, 0].set_xticklabels(hour_labels, rotation=45)
    axes[1, 0].set_xlabel('Время')
    axes[1, 0].set_ylabel('Заявок')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Вторая модель уверенные решения
    axes[1, 1].plot(times, [stat['second_confident'] for stat in batch_stats],
                    'green', linewidth=1.5)
    axes[1, 1].fill_between(times, 0, [stat['second_confident'] for stat in batch_stats],
                            alpha=0.2, color='green')
    axes[1, 1].set_title(f'Уверенные решения: {second_model_name}', fontweight='bold')
    axes[1, 1].set_xticks(hours)
    axes[1, 1].set_xticklabels(hour_labels, rotation=45)
    axes[1, 1].set_xlabel('Время')
    axes[1, 1].set_ylabel('Заявок')
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Ручной разбор от моделей
    axes[2, 0].plot(times, [stat['second_uncertain'] for stat in batch_stats],
                    'orange', linewidth=1.5)
    axes[2, 0].fill_between(times, 0, [stat['second_uncertain'] for stat in batch_stats],
                            alpha=0.2, color='orange')
    axes[2, 0].set_title('Ручной разбор: модели неуверенны', fontweight='bold')
    axes[2, 0].set_xticks(hours)
    axes[2, 0].set_xticklabels(hour_labels, rotation=45)
    axes[2, 0].set_xlabel('Время')
    axes[2, 0].set_ylabel('Заявок')
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Сравнительный график
    axes[2, 1].plot(times, [stat['business_manual'] for stat in batch_stats],
                    'r-', linewidth=1.5, label='Бизнес-правила', alpha=0.7)
    axes[2, 1].plot(times, [stat['second_uncertain'] for stat in batch_stats],
                    'orange', linewidth=1.5, label='Модели неуверенны', alpha=0.7)
    axes[2, 1].set_title('Сравнение источников ручного разбора', fontweight='bold')
    axes[2, 1].set_xticks(hours)
    axes[2, 1].set_xticklabels(hour_labels, rotation=45)
    axes[2, 1].set_xlabel('Время')
    axes[2, 1].set_ylabel('Заявок')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.suptitle('Детальный анализ решений', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return plt

def plot_parameters_history(pid_history, second_model_name="XGBoost", start_time="00:00"):
    """График изменения параметров регулятора"""
    if pid_history is None or pid_history.empty:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    total_minutes = len(pid_history)
    times = range(total_minutes)

    # Метки времени
    hours = range(0, total_minutes, 60)
    hour_labels = [minutes_to_time(m, start_time) for m in hours]

    # 1. Отступы LR
    axes[0].plot(times, pid_history['lr_low'], 'g-', linewidth=2, label='LR Low')
    axes[0].plot(times, pid_history['lr_high'], 'r-', linewidth=2, label='LR High')
    axes[0].set_ylabel('Отступ')
    axes[0].set_title('Отступы Logistic Regression')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(hours)
    axes[0].set_xticklabels(hour_labels, rotation=45)

    # 2. Отступы второй модели
    axes[1].plot(times, pid_history['second_low'], 'g-', linewidth=2, label=f'{second_model_name} Low')
    axes[1].plot(times, pid_history['second_high'], 'r-', linewidth=2, label=f'{second_model_name} High')
    axes[1].set_ylabel('Отступ')
    axes[1].set_title(f'Отступы {second_model_name}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(hours)
    axes[1].set_xticklabels(hour_labels, rotation=45)

    # 3. Ошибка загрузки и выход регулятора
    axes[2].plot(times, pid_history['error_load'], 'b-', label='Error load', alpha=0.7, linewidth=1.5)
    axes[2].plot(times, pid_history['output'], 'r-', label='Output', linewidth=2, alpha=0.7)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_xlabel('Время')
    axes[2].set_ylabel('Значение')
    axes[2].set_title('Ошибка загрузки и выход регулятора')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(hours)
    axes[2].set_xticklabels(hour_labels, rotation=45)

    plt.tight_layout()
    return plt


