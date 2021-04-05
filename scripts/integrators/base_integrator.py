from abc import ABC, abstractmethod


from scripts.dynamic_parameters import SystemDynamicParameters
from scripts.log_config import logger_wraps
from scripts.modeling_parameters import ModelingParameters


class BaseIntegrator(ABC):

    def __init__(
            self,
            dynamic: SystemDynamicParameters,
            model: ModelingParameters,
    ):
        self.dynamic = dynamic
        self.model = model

    @logger_wraps()
    @abstractmethod
    def stage_1(self):
        raise NotImplementedError(
            'Define `stage_1` in'
            f'{self.__class__.__name__}.'
        )

    @logger_wraps()
    @abstractmethod
    def stage_2(self):
        raise NotImplementedError(
            'Define `stage_2` in'
            f'{self.__class__.__name__}.'
        )
