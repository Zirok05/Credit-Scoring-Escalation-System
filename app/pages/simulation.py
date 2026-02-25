import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.data_loader import load_artifacts
from app.simulation.core.traffic_generator import TrafficGenerator
from app.simulation.core.processor import ApplicationProcessor
from app.simulation.controllers.pid import PIDController
from app.simulation.visualization.plots import (
    plot_queue_dynamics,
    plot_specialist_load,
    plot_inflow,
    plot_parameters_history,
    plot_detailed_decisions
)
# ============================================================================
# БЛОК АНИМАЦИИ: Импорт функций для визуализации
# ============================================================================
from app.simulation.visualization.animation import create_simulation_video

# ============================================================================


def minutes_to_time(minutes, start_time="00:00"):
    """Преобразует минуты от старта в строку времени ЧЧ:ММ"""
    start_hour, start_min = map(int, start_time.split(':'))
    total_minutes = start_hour * 60 + start_min + minutes
    hour = (total_minutes // 60) % 24
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def main():
    st.title("📊 Симуляция работы системы")

    # Загрузка артефактов
    PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MODELS_PATH = os.path.join(PROJECT_PATH, 'models/best/train_150/')
    PREPROCESSOR_PATH = os.path.join(PROJECT_PATH, 'preprocessors/')
    TEST_DATA_PATH = os.path.join(PROJECT_PATH, 'datasets/cs-test.csv')

    preprocessor, scaler, models = load_artifacts(MODELS_PATH, PREPROCESSOR_PATH)

    available_models = [name for name in models.keys() if name != 'Logistic Regression']

    st.sidebar.subheader("🤖 Выбор модели")
    second_model_name = st.sidebar.selectbox(
        "Вторая модель для эскалации",
        available_models,
        index=0
    )

    # Параметры симуляции
    st.sidebar.header("⚙️ Параметры")
    # ============================================================================
    # БЛОК АНИМАЦИИ: Ограничение количества специалистов до 400 для таблицы 20x20
    # ============================================================================
    specialists_count = st.sidebar.slider("Количество специалистов (модели)", 10, 400, 100, 10)
    # ============================================================================
    business_specialists_count = st.sidebar.slider("Количество экспертов (бизнес-правила)", 1, 100, 50, 1)

    business_time = st.sidebar.slider("Время обработки бизнес правил(мин)", 5, 30, 15, 5)
    base_time = st.sidebar.slider("Базовое время обработки (мин)", 2, 15, 5)

    target_load = st.sidebar.slider(
        "Целевая загрузка специалистов", 0.5, 1.0, 0.8, 0.05,
        help="0.8 = 80% - оставляем запас на пики")

    st.sidebar.subheader("🎯 Порог одобрения")
    fixed_threshold = st.sidebar.slider(
        "Порог (фиксированный)",
        0.3, 0.7, 0.5, 0.05,
        help="Порог одобрения - стратегический параметр, не меняется PID"
    )

    st.sidebar.subheader("🎯 Начальные отступы (%)")

    lr_low_pct = st.sidebar.slider("LR нижний отступ (% от порога)", 0, 100, 20, 5,
                                   help="% от расстояния между 0 и порогом")
    lr_high_pct = st.sidebar.slider("LR верхний отступ (% от 1-порога)", 0, 100, 20, 5,
                                    help="% от расстояния между порогом и 1")
    second_low_pct = st.sidebar.slider("Вторая модель нижний (%)", 0, 100, 20, 5)
    second_high_pct = st.sidebar.slider("Вторая модель верхний (%)", 0, 100, 20, 5)

    # Преобразуем проценты в абсолютные значения
    init_lr_low = fixed_threshold * lr_low_pct / 100
    init_lr_high = (1 - fixed_threshold) * lr_high_pct / 100
    init_second_low = fixed_threshold * second_low_pct / 100
    init_second_high = (1 - fixed_threshold) * second_high_pct / 100

    # Параметры PID
    st.sidebar.subheader("🎛️ PID регулятор")
    use_pid = st.sidebar.checkbox("Включить PID", value=True)

    if use_pid:
        kp = st.sidebar.slider("P (пропорциональный)", 0.0, 1.0, 0.33)
        ki = st.sidebar.slider("I (интегральный)", 0.0, 1.0, 0.03)
        kd = st.sidebar.slider("D (дифференциальный)", 0.0, 1.0, 0.22)
        w_load = st.sidebar.slider("Вес загрузки", 0.0, 1.0, 0.3)

    # Кнопка запуска
    if st.button("🎬 Запустить симуляцию 24 часа"):
        with st.spinner(f"Загрузка данных и симуляция..."):
            # 1. Загружаем тестовый датасет
            test_df = pd.read_csv(TEST_DATA_PATH)
            if 'SeriousDlqin2yrs' in test_df.columns:
                test_df = test_df.drop(columns=['SeriousDlqin2yrs'])
            test_pool = test_df.to_dict('records')

            # 2. Генерируем распределение заявок по минутам
            current_time = datetime.now()
            start_hour = current_time.hour
            start_minute = current_time.minute

            gen = TrafficGenerator(total_applications=len(test_pool))
            minute_counts = gen.generate_minute_counts(start_hour=start_hour, start_minute=start_minute)

            # Сохраняем для графиков
            st.session_state.start_time = f"{start_hour:02d}:{start_minute:02d}"
            st.session_state.minute_counts = minute_counts

            # 3. Создаём процессор
            processor = ApplicationProcessor(
                lr_model=models['Logistic Regression'],
                second_model=models[second_model_name],
                second_model_name=second_model_name,
                specialists_count=specialists_count,
                business_specialists_count=business_specialists_count,
                base_processing_time=base_time,
                business_processing_time=business_time
            )

            # 4. Создаём PID если нужно
            if use_pid:
                pid = PIDController(
                    init_threshold=fixed_threshold,
                    kp_load=kp, ki_load=ki, kd_load=kd,
                    load_weight=w_load,
                    init_lr_low=init_lr_low,
                    init_lr_high=init_lr_high,
                    init_second_low=init_second_low,
                    init_second_high=init_second_high,
                    target_load=target_load
                )
            else:
                pid = None

            # 5. Симуляция по минутам
            pool_copy = test_pool.copy()
            idx = 0
            progress_bar = st.progress(0)
            n_steps = len(minute_counts)

            # ============================================================================
            # БЛОК АНИМАЦИИ: Сбор данных для кадров
            # ============================================================================
            animation_frames = []  # список для хранения кадров анимации
            # ============================================================================

            for step, n_apps in enumerate(minute_counts):
                # Берём заявки из пула
                batch = pool_copy[idx:idx + n_apps]
                idx += n_apps

                # Получаем текущие параметры
                if pid:
                    margins = pid.get_margins()
                    lr_margins = [margins['lr_low'], margins['lr_high']]
                    second_margins = [margins['second_low'], margins['second_high']]
                    threshold = fixed_threshold
                else:
                    lr_margins = [0.35]
                    second_margins = [0.4]
                    threshold = fixed_threshold

                # Обрабатываем батч
                result = processor.process_batch(
                    batch, preprocessor, scaler,
                    threshold=threshold,
                    lr_margins=lr_margins,
                    second_margins=second_margins,
                    current_time=step
                )

                # Обновляем PID
                if pid:
                    load = result['specialists_busy'] / specialists_count
                    pid.update(load)

                # ============================================================================
                # БЛОК АНИМАЦИИ: Сохраняем кадр каждые 10 минут (чтобы не было 1440 кадров)
                # ============================================================================
                # --- Внутри цикла симуляции в simulation.py ---
                # Записываем КАЖДУЮ минуту для плавности
                if step % 1 == 0 or step == n_steps - 1:
                    specialist_states = processor.specialists.copy()

                    frame_data = {
                        'time': step,
                        'step': step,
                        'time_str': minutes_to_time(step, st.session_state.start_time),
                        'inflow': n_apps,
                        'inflow_history': st.session_state.minute_counts[:step + 1],
                        'load_history': [v / specialists_count for v in processor.stats['specialist_busy'][:step + 1]],
                        'queue': result['queue_size'],
                        'business_queue': result.get('business_queue_size', 0),
                        'load': load if pid else 0,
                        'specialist_states': specialist_states,
                        'cumulative': {
                            'total_processed': processor.stats['total_processed'],
                            'auto_approved': processor.stats['auto_approved'],
                            'auto_declined': processor.stats['auto_declined'],
                            'manual_processed': processor.stats['manual_processed'],
                            'business_manual_processed': processor.stats.get('business_manual_processed', 0)
                        }
                    }
                    animation_frames.append(frame_data)
                # ============================================================================

                # Обновляем прогресс
                progress_bar.progress((step + 1) / n_steps)

            # 6. Сохраняем результаты
            st.session_state.processor = processor
            st.session_state.pid_history = pid.get_history() if pid else None
            st.session_state.simulation_done = True
            st.session_state.batch_stats = processor.batch_stats
            # ============================================================================
            # БЛОК АНИМАЦИИ: Сохраняем кадры в session_state
            # ============================================================================
            st.session_state.animation_frames = animation_frames
            # ============================================================================

    # Отображение результатов
    if st.session_state.get('simulation_done', False):
        st.success("✅ Симуляция завершена!")

        stats = st.session_state.processor.stats

        # Быстрая статистика
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Всего заявок", stats['total_processed'])
        col2.metric("Одобрено авто", stats['auto_approved'])
        col3.metric("Отказ авто", stats['auto_declined'])
        col4.metric("Ручной разбор", stats['manual_processed'])
        manual_rate = stats['manual_sent'] / stats['total_processed'] * 100 if stats['total_processed'] > 0 else 0
        col5.metric("Ручной разбор %", f"{manual_rate:.1f}%")

        # Графики
        st.subheader("📈 Графики")

        # Очереди
        st.pyplot(plot_queue_dynamics(
            queue_history=stats['queue_history'],
            business_queue_history=stats.get('business_queue_history'),
            start_time=st.session_state.get('start_time', '00:00')
        ))
        plt.close()

        # Загрузка специалистов
        st.pyplot(plot_specialist_load(
            specialist_busy_history=stats['specialist_busy'],
            specialists_count=specialists_count,
            start_time=st.session_state.get('start_time', '00:00')
        ))
        plt.close()
        st.pyplot(plot_inflow(
            minute_counts=st.session_state.minute_counts,
            start_time=st.session_state.get('start_time', '00:00')
        ))
        plt.close()
        # Детальный анализ решений
        st.pyplot(plot_detailed_decisions(
            batch_stats=st.session_state.batch_stats,
            second_model_name=second_model_name,
            start_time=st.session_state.get('start_time', '00:00')
        ))
        plt.close()
        # Параметры PID
        st.pyplot(plot_parameters_history(
            pid_history=st.session_state.pid_history,
            second_model_name=second_model_name,
            start_time=st.session_state.get('start_time', '00:00')
        ))
        plt.close()
        # ============================================================================
        # ГЕНЕРАЦИЯ ВИДЕО
        # ============================================================================
        if st.session_state.get('animation_frames'):
            st.divider()
            st.subheader("🎥 Настройки видео-отчета")

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                # Слайдер для шага кадров (среза)
                v_step = st.slider("Шаг кадров (1 = каждая минута)", 1, 30, 1,
                                   help="Чем меньше шаг, тем плавнее видео, но дольше рендеринг")
            with col_v2:
                # Слайдер для FPS
                v_fps = st.slider("Скорость видео (FPS)", 10, 60, 24,
                                  help="Количество кадров в секунду")

            if st.button("🎬 Сгенерировать видео", type="primary", use_container_width=True):
                with st.spinner("Рендеринг видео..."):

                    # Используем выбранные в слайдерах параметры
                    video_path = create_simulation_video(
                        st.session_state.animation_frames[::v_step],
                        specialists_count,
                        second_model_name,
                        fps=v_fps  # Передаем FPS в функцию
                    )
                    st.video(video_path)
                    st.success("✅ Видео готово! Вы можете его скачать или перематывать.")

        st.write("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🏠 На главную", use_container_width=True):
                st.switch_page("main.py")


if __name__ == "__main__":
    main()