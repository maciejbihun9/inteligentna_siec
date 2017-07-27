from src.data_manager import DataManager
from numpy import *
from src.math_res import MathResources
from src.normalizer import Normalizer
from src.norm_type import NormType
from sklearn import linear_model, datasets, model_selection
from src.credibility import Credibility
from numpy.linalg import *
from src.learn_methods import LearnMethod
from src.error_msg import ErrorMsg

"""
* Implement it using mathematical patterns
"""
class LogRes:

    """
    * set classes index column
    - the rest of columns indexes will tract as data
    """

    def __init__(self, train_inputs: ndarray, train_target: ndarray, test_inputs: ndarray, test_target: ndarray):
        self.train_inputs = train_inputs
        self.train_target = train_target
        self.test_inputs = test_inputs
        self.test_target = test_target
        self.beta = array([])

    def least_square_fit(self):
        """
        Method used for linear regression usually.
        For logistic regression not work properly.(Low value of effeciency)
        :return:
        """
        self.train_inputs = mat(self.train_inputs)
        self.train_target = mat(self.train_target)
        self.beta = inv(self.train_inputs.T * self.train_inputs) * self.train_inputs.T * self.train_target.T

    def grad_desc_fit(self, error):
        """
                In this class we perform only computations aspect. Prepare data should be moved to data_manager.
                :param inputs:
                :param target:
                :param categorical_mask:
                :return:
                """
        m, n = shape(self.train_inputs)
        alfa = 0.5
        rate = 0.01
        beta = MathResources.get_init_beta(n)
        prev_beta = beta
        prev_cost = inf
        while prev_cost > error:
            beta = MathResources.log_reg(self.train_inputs, self.train_target, beta, alfa)
            cost = MathResources.get_cost_func_value(self.train_inputs, self.train_target, beta)
            if prev_cost < cost:
                beta = copy(prev_beta)
                alfa -= alfa * rate
                margin = cost - prev_cost
                print("prev cost: {}".format(prev_cost))
                print("Alfa : {}".format(alfa))
                print("Cost - prev_cost margin: {}".format(margin))
                continue
            prev_beta = copy(beta)
            prev_cost = copy(cost)
            print("Cost : {}".format(cost))

        print("Final cost : {}".format(prev_cost))
        print("Final beta: {}".format(beta))
        self.beta = beta



    def adagrad_fit(self, error=0.001):
        """
        In this class we perform only computations aspect. Prepare data should be moved to data_manager.
        :param inputs:
        :param target:
        :param categorical_mask:
        :return:
        """
        m, n = shape(self.train_inputs)

        alfa = 0.3
        beta = MathResources.get_init_beta(n)
        prev_cost = inf
        grads = []
        while prev_cost > error:
            grad = MathResources.get_grad(self.train_inputs, self.train_target, beta)
            grads.append(grad)
            delta = LearnMethod.adagrad(alfa, array(grad), array(grads))
            beta += delta
            # beta = MathResources.log_reg(self.train_inputs, self.train_target, new_beta, alfa)
            cost = MathResources.get_cost_func_value(self.train_inputs, self.train_target, beta)
            if prev_cost < cost:
                print("Final cost: {}".format(cost))
                break
            prev_cost = copy(cost)
            print("Cost: {}".format(cost))
        print("Final cost : {}".format(prev_cost))
        print("Final beta: {}".format(beta))
        self.beta = beta


    def adadelta_fit(self):
        self.beta = LearnMethod.adadelta(self.train_inputs, self.train_target, 0.4)


    def fit_test(self) -> float:
        """
        Fit trained betas with examples.
        Return the results as lit of tuples.
        :return:
        """
        if len(self.beta) == 0:
            raise ValueError(ErrorMsg.BETA_NOT_INITIALIZED)

        results = []
        for i in range(len(self.test_inputs)):
            result = sum(self.test_inputs[i] * self.beta)
            result = MathResources.get_log_res_func(result)
            target = self.test_target[i]
            results.append((result, target))
            print("res : {} and tar: {}".format(result, target))
        credibility = Credibility(results)
        specifity = credibility.get_specificity()
        accuracy = credibility.get_accuracy()
        precision = credibility.get_precision()
        f_score = credibility.get_f_score()
        sensitivity = credibility.get_sensitivity()
        print("results")


    def lib_fit(self) -> list:
        return []



