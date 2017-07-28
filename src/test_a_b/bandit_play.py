
import matplotlib.pyplot as plt
from src.test_a_b.bandit import Bandit
import numpy as np

def sample_distributions_and_choose(bandit_params):
    # W tej metodzie używana jest metoda beta.rvs
    # do pobierania losowej zmiennej z każdegorozkładu beta
    # Metoda zwraca indeks listy uzskany na podstawie próbki
    # o największej wartości. Ten indeks wyznacza drążek,
    # który zostanie pociągnięty jako następny w eksperymencie z wielorękim bandytą.
    sample_array = [beta.rvs(param[0], param[1], size=1)[0] for param in bandit_params]
    return np.argmax(sample_array)

def run_single_regret(bandit_list, bandit_params, plays):
    """
    :param bandit_params: machine list with list of dist params
    :param plays: Number of shoots to take
    :return:
    """
    sum_probs_chosen = 0
    opt = np.zeros(plays)
    chosen = np.zeros(plays)
    bandit_probs = [x.get_prob() for x in bandit_list]
    opt_solution = max(bandit_probs)
    for i in range(0, plays):
        index = sample_distributions_and_choose(bandit_params)
        sum_probs_chosen += bandit_probs[index]
        if (bandit_list[index].pull_handle()):
            bandit_params[index] = \
                (bandit_params[index][0] + 1, bandit_params[index][1])
        else:
            bandit_params[index] = \
                (bandit_params[index][0], bandit_params[index][1] + 1)
        opt[i] = (i + 1) * opt_solution
        chosen[i] = sum_probs_chosen
    regret_total = map(sub, opt, chosen)
    return regret_total

bandit_list = [Bandit(0.1),Bandit(0.3),Bandit(0.8)]
bandit_params = [(1,1),(1,1),(1,1)]
x = np.linspace(0,1, 100)
plt.plot(x,
beta.pdf(x, bandit_params[0][0], bandit_params[0][1]),
'-r*',
alpha=0.6,
label='Maszyna 1')
plt.plot(x,
beta.pdf(x, bandit_params[1][0], bandit_params[1][1]),
'-b+',
alpha=0.6,
label='Maszyna 2')
plt.plot(x,
beta.pdf(x, bandit_params[2][0], bandit_params[2][1]),
'-go',
alpha=0.6,
label='Maszyna 3')
plt.legend()
plt.xlabel("Prawdopodobieństwo wypłaty")
plt.ylabel("Gęstość prawdopodobieństwa przekonań")
plt.show()