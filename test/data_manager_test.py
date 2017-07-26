

import unittest
from src.data_manager import DataManager
from numpy import *

class DataManagerTest(unittest.TestCase):

    def setUp(self):
        url = '../resources/50k.txt'
        data = DataManager.load_data(url, False, False)
        self.data = array(data)

    def test_data_filter(self):
        filter_sign = '?'
        filtered_data = DataManager.data_filter(self.data, filter_sign)
        self.assertFalse(filter_sign in filtered_data)
