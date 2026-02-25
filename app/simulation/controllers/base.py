from abc import ABC, abstractmethod


class BaseController(ABC):
    """Базовый класс для всех контроллеров"""

    def __init__(self, name="Base"):
        self.name = name
        self.history = []

    @abstractmethod
    def update(self, current_state, target_state):
        """
        Рассчитывает новые параметры управления

        Параметры:
        - current_state: текущее состояние системы (очередь, загрузка)
        - target_state: целевое состояние

        Возвращает:
        - новые пороги и отступы
        """
        pass

    def get_margins(self, hour=None):
        """Возвращает текущие отступы для LR и второй модели"""
        pass