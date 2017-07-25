
import unittest
from numpy import *
from src.learn_methods import LearnMethod

class LearnMethodsTest(unittest.TestCast):


    def test_adagrad(self):
        alfa = 0.03
        grad = array([1, 3, 4])
        prev_grads = array([[2, 3, 1], [4, 3, 5], [1.5, 2, 1]])
        delta = LearnMethod.adagrad(alfa, grad, prev_grads)
        print(delta)


