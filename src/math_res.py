
from numpy import *
import math

class MathResources:

    @staticmethod
    def get_hypo_value(x_values: ndarray, beta_values: ndarray) -> float:
        """
        :param x_values: (1, n) range array of x values
        :param beta_values: (1, n) range array of beta values
        :return:
        """
        if shape(x_values) != shape(beta_values):
            raise ValueError("Arrays shape not equal")
        return sum(beta_values * x_values)


    @staticmethod
    def get_log_res_func(hypo_value: float):
        """
        hypo_value can not have optional values.
        :param hypo_value: hypothesis function result value
        :param k: number of
        :return:
        """
        value = 1 / (1 + exp(-2 * hypo_value))
        if math.isnan(value):
            print("Error")
        return value

    @staticmethod
    def get_init_beta(num: int) -> ndarray:
        """
        :param num: Number of betas to return
        :return: ndarray of float values
        """
        return (2.5 * random.randn(1, num) + 3)[0]


    @staticmethod
    def get_cost_func_value(data: ndarray, target: ndarray, beta: ndarray):
        m, n = shape(data)
        suma = 0
        for i in range(m):
            hypo_value = MathResources.get_hypo_value(data[i], beta)
            # suma += target[i] * log(hypo_value) + (1 - target[i]) * log(1 - hypo_value)
            if hypo_value <= 0:
                suma += (1 - target[i]) * log(1 - hypo_value)
            else:
                suma += target[i] * log(hypo_value)
        return (1/m) * suma

    @staticmethod
    def log_reg(data: ndarray, target: ndarray, beta: ndarray, alfa) -> ndarray:
        """
        Learn can can not be updated during one column learning.
        We can return list of grads with beta
        :param data:
        :param target:
        :param beta:
        :param alfa:
        :return:
        """
        m, n = shape(data)
        if len(data) != len(target):
            raise ValueError("ndarrays shape not equal!")
        # for each param
        for j in range(n):
            grad = 0
            for m in range(m):
                est_prop = MathResources.get_log_res_func(MathResources.get_hypo_value(data[m], beta))
                if math.isnan(est_prop):
                    est_prop = 1

                grad += (est_prop - target[m]) * data[m, j]
            beta[j] = beta[j] - (alfa * (grad / m))
        return beta

    # compute gradient
    @staticmethod
    def get_grad(data: ndarray, target: ndarray, beta: ndarray) -> ndarray:
        """
        Compute gradient value only for one column
        :param data:
        :param target:
        :param beta:
        :return: ndarray of params gradients
        """
        m, n = shape(data)
        if len(data) != len(target):
            raise ValueError("ndarrays shape not equal!")
        # for each column(param)
        grads = array([0] * n)
        for j in range(n):
            grad = 0
            for m in range(m):
                est_prop = MathResources.get_log_res_func(MathResources.get_hypo_value(data[m], beta))
                if math.isnan(est_prop):
                    est_prop = 1
                grad += (est_prop - target[m]) * data[m, j]
            grads[j] = grad
        return grads




