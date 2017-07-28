

"""
 * wsp. konwersji - ilość zakupów / ilości wyświetleń
 jest to również zmienna losowa o rozkładzie normalnym,


 * Dobra metoda do testowania skutków dziłania
 *
 * Test A/B prowadzi do przetestowania jak sprawdzi się nowe rozwiązanie,
 * Zmieniana może być tylko kolejność elementów w grupie,
 * grupy nie mogą różnić się błędem systematycznycm,
 * Experimental Hypothesis tells that we have variables higher that in controll group
 * Remember that bigger set always gives you better and smaller estimation of the standard error

 * Aby ustalić czy nowa wersja będzie bardziej dochodowa trzeba przeprowadzić test poziomu ufności

 * It is though to get good results in production because lack of data items in the dataset.
 * items are linked with bad groups
 * there is really small difference between results of our groups.
 * It is though to get good results

 An example:

"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
n_experiment = 10000
n_control = 10000
p_experiment= 0.002
p_control = 0.001
se_experiment_sq = p_experiment*(1-p_experiment) / n_experiment
se_control_sq = p_control*(1-p_control) / n_control
Z = (p_experiment-p_control)/math.sqrt(se_experiment_sq+se_control_sq)
print("Z value : {}".format(Z))

