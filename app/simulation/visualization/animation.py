import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tempfile
import numpy as np
import os
from matplotlib.animation import FFMpegWriter

def minutes_to_time(minutes, start_time="00:00"):
    start_hour, start_min = map(int, start_time.split(':'))
    total_minutes = start_hour * 60 + start_min + minutes
    hour = (total_minutes // 60) % 24
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"

def create_simulation_video(frames, specialists_count, second_model_name, fps=24):
    if not frames:
        return None

    # Настройка стиля
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor='#f8f9fa')
    plt.subplots_adjust(hspace=0.4, wspace=0.25)
    plt.close()

    def update(i):
        data = frames[i]
        for ax in axes.flatten():
            ax.clear()
            ax.set_facecolor('white')

        # 1. ДИНАМИКА ПОТОКА (Локализация)
        y_inflow = data['inflow_history']
        axes[0, 0].fill_between(range(len(y_inflow)), y_inflow, color='#4361ee', alpha=0.3)
        axes[0, 0].plot(range(len(y_inflow)), y_inflow, color='#4361ee', linewidth=2)
        axes[0, 0].set_xlim(0, 1440)  # Фиксация оси времени
        axes[0, 0].set_title("ДИНАМИКА ПОТОКА (заявок/мин)", fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel("Минуты симуляции")

        # 2. ЗАГРУЗКА СИСТЕМЫ
        y_load = [v * 100 for v in data['load_history']]
        axes[0, 1].fill_between(range(len(y_load)), y_load, color='#4cc9f0', alpha=0.3)
        axes[0, 1].plot(range(len(y_load)), y_load, color='#4cc9f0', linewidth=2)
        axes[0, 1].axhline(y=80, color='#f72585', linestyle='--', alpha=0.6)
        axes[0, 1].set_xlim(0, 1440)
        axes[0, 1].set_ylim(0, 110)
        axes[0, 1].set_title(f"ЗАГРУЖЕННОСТЬ СПЕЦИАЛИСТОВ %: {y_load[-1]:.1f}%", fontsize=12, fontweight='bold')

        # 3. HEATMAP И ЛЕГЕНДА
        states = np.array(data['specialist_states'])
        cols = 20
        rows = int(np.ceil(specialists_count / cols))
        z = np.zeros((rows, cols))
        for idx, val in enumerate(states[:rows * cols]):
            z[idx // cols, idx % cols] = val

        im = axes[1, 0].imshow(z, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=10)
        axes[1, 0].set_title(f"МОНИТОРИНГ: {specialists_count} СПЕЦИАЛИСТОВ", fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        # Добавляем текстовую легенду под хитмапом
        legend_text = "Цвета: Зеленый (Свободен) → Желтый (3-5 мин) → Красный (8+ мин)"
        axes[1, 0].text(0.5, -0.1, legend_text, ha='center', transform=axes[1, 0].transAxes, fontsize=10)

        # 4. РАЗДЕЛЕННЫЕ ОЧЕРЕДИ И СТАТИСТИКА
        ax_stat = axes[1, 1]
        ax_stat.clear()
        ax_stat.axis('off')

        # Цвета для очередей (краснеют, если очередь > 50)
        q_mod_color = '#991b1b' if data['queue'] > 50 else '#166534'
        q_biz_color = '#991b1b' if data.get('business_queue', 0) > 50 else '#1e293b'

        # Две надписи очередей сверху
        ax_stat.text(0.25, 0.9, "ОЧЕРЕДЬ\n(МОДЕЛИ)", fontsize=10, ha='center', fontweight='bold')
        ax_stat.text(0.25, 0.78, f"{data['queue']}", fontsize=26, ha='center', fontweight='bold', color=q_mod_color)

        ax_stat.text(0.75, 0.9, "ОЧЕРЕДЬ\n(БИЗНЕС ПРАВИЛА)", fontsize=10, ha='center', fontweight='bold')
        ax_stat.text(0.75, 0.78, f"{data.get('business_queue', 0)}", fontsize=26, ha='center', fontweight='bold',
                     color=q_biz_color)

        # Сводная таблица
        cum = data['cumulative']
        stats_text = (
            f"Итоговые показатели к {data['time_str']}\n"
            f"--------------------------------------\n"
            f"ОБРАБОТАНО ВСЕГО:                 {cum['total_processed']}\n"
            f"Авто-одобрено:                    {cum['auto_approved']}\n"
            f"Авто-отказы:                      {cum['auto_declined']}\n"
            f"Ручной разбор (модель):           {cum['manual_processed']}\n"
            f"Ручной разбор (бизнес правила):   {cum['business_manual_processed']}\n"
            f"--------------------------------------\n"
            f"Используемая модель: {second_model_name}"
        )

        ax_stat.text(0.5, 0.3, stats_text, fontsize=10, fontfamily='monospace',
                     ha='center', va='center', transform=ax_stat.transAxes,
                     bbox=dict(facecolor='#f8f9fa', alpha=1, boxstyle='round,pad=1', edgecolor='#dee2e6'))

        return axes.flatten()

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

    writer = animation.FFMpegWriter(fps=fps, bitrate=2000, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    ani.save(tmp_file.name, writer=writer)
    return tmp_file.name