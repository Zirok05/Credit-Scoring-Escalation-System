import numpy as np
import pandas as pd
from .base import BaseController
class PIDController(BaseController):
    """PID-регулятор для управления отступами на основе загрузки специалистов"""

    def __init__(self, name="PID",
                 kp_load=0.1, ki_load=0.01, kd_load=0.05,
                 load_weight=1.0,
                 # Начальные значения параметров
                 init_threshold=0.5,
                 init_lr_low=0.3, init_lr_high=0.4,
                 init_second_low=0.35, init_second_high=0.45,
                 target_load=0.8):
        super().__init__(name)

        # Коэффициенты PID для загрузки
        self.kp_load = kp_load
        self.ki_load = ki_load
        self.kd_load = kd_load

        self.load_weight = load_weight
        self.target_load = target_load

        # Состояния PID
        self.prev_error_load = 0
        self.integral_load = 0

        # Начальные параметры
        self.init_threshold = init_threshold
        self.init_lr_low = init_lr_low
        self.init_lr_high = init_lr_high
        self.init_second_low = init_second_low
        self.init_second_high = init_second_high

        # Текущие параметры (отступы)
        self.threshold = init_threshold
        self.lr_low = init_lr_low
        self.lr_high = init_lr_high
        self.second_low = init_second_low
        self.second_high = init_second_high

        # Границы отступов
        self.bounds = {
            'lr_low': (0.05, self.threshold - 0.05),
            'lr_high': (0.05, 1 - self.threshold - 0.05),
            'second_low': (0.05, self.threshold - 0.05),
            'second_high': (0.05, 1 - self.threshold - 0.05)
        }

        # Ограничение интеграла
        self.integral_limit = 1.0

    def update(self, current_load):
        """
        current_load: текущая загрузка специалистов (0-1)
        """
        # Ошибка по загрузке
        error_load = self.target_load - current_load

        # PID для загрузки
        P_load = self.kp_load * error_load
        self.integral_load += error_load
        self.integral_load = np.clip(self.integral_load, -self.integral_limit, self.integral_limit)
        I_load = self.ki_load * self.integral_load
        D_load = self.kd_load * (error_load - self.prev_error_load)
        self.prev_error_load = error_load

        # Выход регулятора
        output_load = P_load + I_load + D_load
        output = self.load_weight * output_load

        # Адаптируем отступы
        self._update_parameters(output)

        # Сохраняем историю
        self.history.append({
            'time': len(self.history),
            'error_load': error_load,
            'output': output,
            'threshold': self.threshold,
            'lr_low': self.lr_low,
            'lr_high': self.lr_high,
            'second_low': self.second_low,
            'second_high': self.second_high,
            'load': current_load,
        })

        return self.get_margins()

    def _update_parameters(self, output):
        """Обновляет отступы на основе выхода регулятора"""
        delta = output * 0.1
        self.lr_low = np.clip(
            self.lr_low + delta,
            self.bounds['lr_low'][0],
            self.bounds['lr_low'][1]
        )
        self.lr_high = np.clip(
            self.lr_high + delta,
            self.bounds['lr_high'][0],
            self.bounds['lr_high'][1]
        )
        self.second_low = np.clip(
            self.second_low + delta,
            self.bounds['second_low'][0],
            self.bounds['second_low'][1]
        )
        self.second_high = np.clip(
            self.second_high + delta,
            self.bounds['second_high'][0],
            self.bounds['second_high'][1]
        )

    def get_margins(self, hour=None):
        """Возвращает текущие отступы"""
        return {
            'lr_low': self.lr_low,
            'lr_high': self.lr_high,
            'second_low': self.second_low,
            'second_high': self.second_high
        }

    def get_history(self):
        """Возвращает историю для визуализации"""
        return pd.DataFrame(self.history)