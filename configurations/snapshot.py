from configurations.static_system import StaticSystem


class Snapshot(StaticSystem):

    def __init__(self, time: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.time = time

    @property
    def time(self) -> float:
        return self.__time

    @time.setter
    def time(self, time: float) -> None:
        self.__time = time
