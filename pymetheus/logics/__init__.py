from enum import Enum, unique

from pymetheus.logics.interfaces import *
from pymetheus.logics.lukasiewicz_logic import LukasiewiczLogic


@unique
class LogicEnum(Enum):
    """
    Enumerates the logics available for LogicNet
    """
    LUKASIEWICZ = LukasiewiczLogic()

    @classmethod
    def default(cls) -> 'LogicEnum':
        """
        :return: the default logic
        :rtype: LogicEnum
        """
        return cls.LUKASIEWICZ
