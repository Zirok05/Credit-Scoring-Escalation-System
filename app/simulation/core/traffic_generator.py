import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class TrafficGenerator:
    def __init__(self, total_applications=101503, random_seed=42):
        self.total = total_applications
        np.random.seed(random_seed)

        # Параметры интенсивности
        self.intensity_params = {
            'background': 0.1,
            'day_center': 13, 'day_amplitude': 0.9, 'day_width': 2.5,  # день поуже
            'evening_center': 19.5, 'evening_amplitude': 1.3, 'evening_width': 2.2,  # вечер пораньше и пошире
            'afternoon_dip_center': 15.5, 'afternoon_dip_strength': 0.3, 'afternoon_dip_width': 1.5,  # провал после обеда
            'noise_level': 0.1
        }

    def _time_to_hours(self, time_tuple):
        """Переводит (часы, минуты) в часы с дробной частью"""
        return time_tuple[0] + time_tuple[1] / 60

    def loan_intensity_periodic(self, t, impulses=None):
        """
        Функция интенсивности
        t: время в часах (может быть дробным)
        impulses: список словарей вида
            [{'time': (16, 37), 'strength': 2.0}, ...]  # время как (часы, минуты)
        """
        t_cycle = t % 24

        bg = self.intensity_params['background']

        # Утренне-дневной пик (13:00)
        day = self.intensity_params['day_amplitude'] * np.exp(
            -(t_cycle - self.intensity_params['day_center']) ** 2 /
            (2 * self.intensity_params['day_width'] ** 2)
        )

        # Вечерний пик (19:30)
        evening_diff = np.minimum(
            np.abs(t_cycle - self.intensity_params['evening_center']),
            np.abs(t_cycle - self.intensity_params['evening_center'] + 24)
        )
        evening = self.intensity_params['evening_amplitude'] * np.exp(
            -(evening_diff) ** 2 / (2 * self.intensity_params['evening_width'] ** 2)
        )

        # Провал после обеда (15:30)
        dip_diff = np.minimum(
            np.abs(t_cycle - self.intensity_params['afternoon_dip_center']),
            np.abs(t_cycle - self.intensity_params['afternoon_dip_center'] + 24)
        )
        dip = -self.intensity_params['afternoon_dip_strength'] * np.exp(
            -(dip_diff) ** 2 / (2 * self.intensity_params['afternoon_dip_width'] ** 2)
        )

        intensity = bg + day + evening + dip
        intensity = np.maximum(intensity, 0.05)  # не ниже минимума

        # Шум
        if self.intensity_params['noise_level'] > 0:
            noise = 1.0 + np.random.uniform(
                -self.intensity_params['noise_level'],
                self.intensity_params['noise_level']
            )
            intensity *= noise

        # Импульсы
        if impulses:
            for imp in impulses:
                imp_time = self._time_to_hours(imp['time']) % 24
                # Используем гауссиану для плавного импульса (ширина ~30 минут)
                imp_diff = np.minimum(
                    np.abs(t_cycle - imp_time),
                    np.abs(t_cycle - imp_time + 24)
                )
                imp_factor = 1.0 + imp['strength'] * np.exp(-(imp_diff) ** 2 / (2 * 0.25 ** 2))
                intensity *= imp_factor

        return intensity

    def generate_minute_counts(self, start_hour=None, start_minute=0, impulses=None):
        """
        Возвращает массив количества заявок на каждую минуту (1440 значений)

        start_hour: час старта (по умолчанию текущий)
        start_minute: минута старта
        impulses: список импульсов, например:
            [{'time': (5, 30), 'strength': 2.0}, ...]  # импульс в 5:30 силой 2.0
        """
        if start_hour is None:
            now = datetime.now()
            start_hour = now.hour
            start_minute = now.minute

        start_time = start_hour + start_minute / 60

        # Массив минут (от start_time до start_time + 24)
        minutes = np.arange(0, 24, 1 / 60)
        intensity_values = np.array([
            self.loan_intensity_periodic(start_time + m, impulses)
            for m in minutes
        ])

        total_intensity = np.sum(intensity_values)
        scale_factor = self.total / total_intensity

        minute_counts = np.floor(intensity_values * scale_factor).astype(int)

        # Распределяем остаток (чтоб точно сошлось общее число)
        total_assigned = np.sum(minute_counts)
        if total_assigned < self.total:
            remainder = self.total - total_assigned
            top_minutes = np.argsort(intensity_values)[-remainder:]
            minute_counts[top_minutes] += 1

        return minute_counts

    def generate_hourly_counts(self, start_hour=None, start_minute=0, impulses=None):
        """
        Возвращает массив количества заявок по часам (24 значения)
        """
        minute_counts = self.generate_minute_counts(start_hour, start_minute, impulses)
        hourly_counts = [np.sum(minute_counts[i * 60:(i + 1) * 60]) for i in range(24)]
        return hourly_counts

    def generate_random_impulses(self, n_impulses=1, min_strength=1.5, max_strength=3.0):
        """
        Генерирует случайные импульсы
        """
        impulses = []
        for _ in range(n_impulses):
            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 60)
            strength = np.random.uniform(min_strength, max_strength)
            impulses.append({'time': (hour, minute), 'strength': strength})
        return impulses

    def plot_distribution(self, start_hour=None, start_minute=0, impulses=None):
        """Строит график распределения заявок по часам"""
        hourly_counts = self.generate_hourly_counts(start_hour, start_minute, impulses)

        if start_hour is None:
            start_hour = datetime.now().hour

        hours = [(start_hour + i) % 24 for i in range(24)]
        sorted_pairs = sorted(zip(hours, hourly_counts))
        hours_sorted, counts_sorted = zip(*sorted_pairs)

        plt.figure(figsize=(14, 6))

        # Цвета в зависимости от времени суток
        colors = []
        for h in hours_sorted:
            if 0 <= h <= 5:
                colors.append('#2c3e50')  # ночь
            elif 6 <= h <= 11:
                colors.append('#3498db')  # утро
            elif 12 <= h <= 16:
                colors.append('#f39c12')  # день (с провалом)
            else:
                colors.append('#e67e22')  # вечер

        bars = plt.bar([str(h) for h in hours_sorted], counts_sorted,
                       alpha=0.8, color=colors, edgecolor='black', linewidth=1)

        # Средняя линия
        mean_val = np.mean(counts_sorted)
        plt.axhline(y=mean_val, color='red', linestyle='--',
                    alpha=0.7, linewidth=2, label=f'Среднее: {mean_val:.0f}')

        # Отметим импульсы на графике
        if impulses:
            for imp in impulses:
                imp_hours = self._time_to_hours(imp['time']) % 24
                # Найдём ближайший час
                closest_hour = min(hours_sorted, key=lambda x: abs(x - imp_hours))
                idx = list(hours_sorted).index(closest_hour)
                plt.plot(idx, counts_sorted[idx], 'g*', markersize=15,
                         label=f'Импульс {imp["strength"]:.1f}x' if idx == 0 else '')

        # Отметим провал после обеда
        dip_idx = [i for i, h in enumerate(hours_sorted) if 14 <= h <= 16]
        if dip_idx:
            plt.axvspan(dip_idx[0] - 0.4, dip_idx[-1] + 0.4, alpha=0.2, color='gray',
                        label='Послеобеденный спад')

        plt.xlabel('Час', fontsize=12)
        plt.ylabel('Количество заявок', fontsize=12)
        plt.title(f'Распределение заявок по часам (старт в {start_hour:02d}:{start_minute:02d})',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Статистика
        print("\n📊 Статистика распределения:")
        print(f"   Всего заявок: {sum(counts_sorted)}")
        print(f"   Среднее: {mean_val:.0f} заявок/час")
        print(f"   Максимум: {max(counts_sorted)} заявок")
        print(f"   Минимум: {min(counts_sorted)} заявок")

        return hours_sorted, counts_sorted


# Пример использования
# if __name__ == "__main__":
#
#     gen = TrafficGenerator(total_applications=110000)
#
#     # Без импульсов
#     print("Без импульсов:")
#     counts = gen.generate_minute_counts(start_hour=17)
#     print(f"Всего минут: {len(counts)}")
#     print(f"Всего заявок: {sum(counts)}")
#
#     # С импульсом в 5:30 утра
#     impulses = [{'time': (5, 30), 'strength': 2.0}]
#     print("\nС импульсом в 5:30:")
#     counts = gen.generate_minute_counts(start_hour=17, impulses=impulses)
#
#     # График
#     gen.plot_distribution(start_hour=17, impulses=impulses)
#
#     # Случайные импульсы
#     random_impulses = gen.generate_random_impulses(n_impulses=2)
#     print("\nСлучайные импульсы:", random_impulses)
#     gen.plot_distribution(start_hour=17, impulses=random_impulses)