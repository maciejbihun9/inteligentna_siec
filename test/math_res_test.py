
import unittest
from src.math_res import MathResources
from numpy import *

class MathResTest(unittest.TestCase):

    # TESTED
    def test_div_arr(self):
        arr1 = array([[3, 0, 1, 2], [0, 0, 1, 2]])
        arr2 = array([2, 0, 1, 2])

        value = MathResources.div_arr(arr1, arr2)
        exp = array([[ 1.5,  0.,   1.,   1. ], [ 0.,   0.,   1.,   1. ]])
        self.assertTrue(array_equal(value, exp))

        # check if method works also with numbers
        value = MathResources.div_arr(4, arr1)
        print(value)

        value = MathResources.div_arr(arr1, 3)
        print(value)

    def test_get_init_beta(self):
        """
        Test returned beta values
        :return: None
        """
        betas = MathResources.get_init_beta(10)
        print(betas)


    def test_multiplying(self):
        a = 0.012
        b = 0.243
        result = a * b
        print(result)