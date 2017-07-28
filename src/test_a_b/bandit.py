
from numpy import *

class Bandit:

    def __init__(self, probability):
        self.probability = probability

    def pull_handle(self):
        if random.random() < self.probability:
            return 1
        else:
            return 0

    def get_prob(self):
        return self.probability



