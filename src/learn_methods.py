from numpy import *
from src.math_res import MathResources

class LearnMethod:
    """
    This class contains learn methods for machine learning and neural nets systems.
    Each of this methods returns learned parameters.
    """

    @staticmethod
    def adadelta(input: ndarray, target: ndarray, rate: float, epsilon = 0.00001, ro = 0.95, error = 0.0001) -> ndarray:
        """
        We need initialized x1 parameter
        This learning method do not need any initial learning rate constant.
        Method effectively updates parameters with learning rate based on process gradients.
        :return: ndarray of learned params
        """

        prev_Eg2 = 0

        prev_E_delta = 0

        prev_rms_delta = 0
        # accumulate gradient

        m, n = shape(input)
        beta = MathResources.get_init_beta(n)
        newbeta = 0
        curr_cost = inf
        # update params until cost is low enough
        while curr_cost > error:
            curr_grad = MathResources.get_grad(input, target, beta)

            curr_Eg2 = ro * prev_Eg2 + (1 - ro) * pow(curr_grad, 2)
            # print("curr_Eg2: {}".format(curr_Eg2))

            curr_rms_Eg2 = sqrt(curr_Eg2 + epsilon)
            # print("curr_rms_Eg2: {}".format(curr_rms_Eg2))

            curr_delta = -(prev_rms_delta / curr_rms_Eg2) * curr_grad

            curr_E_delta = ro * prev_E_delta + (1 - ro) * pow(curr_delta, 2)

            curr_rms_delta = sqrt(curr_E_delta)

            beta += curr_delta

            curr_cost = MathResources.get_cost_func_value(input, target, beta)

            prev_E_delta = curr_E_delta

            prev_Eg2 = curr_Eg2

            prev_rms_delta = curr_rms_delta

            print("Delta: {}".format(beta[0]))
            print("Current cost : {}".format(curr_cost))
        return beta

    @staticmethod
    def adagrad(alfa: float, cur_grad: ndarray, prev_grads: ndarray) -> ndarray:
        """
        Method should return ndarray of params delta.
        Method has been checked against zero values.
        :param alfa: constant learning rate
        :param cur_grad: current gradient ndarray
        :param prev_grads: ndarray of previous gradients
        :return: ndarray of params delta
        """
        pow_grads = pow(prev_grads, 2)
        sum_grads = sum(pow_grads, axis = 0)
        coeff = MathResources.div_arr(alfa, (sqrt(sum_grads)))
        length = len(coeff)
        results = array([0.0] * length)
        for i in range(len(coeff)):
            val1 = round(float(coeff[i]), 4)
            val2 = round(float(cur_grad[i]), 4)
            results[i] = val1 * val2
        return -results
