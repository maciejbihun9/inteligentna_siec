
from numpy import *


class MathOper:

    @staticmethod
    def values_between(data_array: ndarray, min: float, max: float) -> ndarray:
        """
        Get array with values that indicates weather are between min and max
        :param data_array: data array to check(Size does not metter)
        :param min: Minimal value
        :param max: Maximal value
        :return: True/False ndarray
        """
        return logical_and(data_array >= min, data_array <= max)