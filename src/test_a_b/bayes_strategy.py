# right from the start we have to wathc out on our knowledge

# wykorzystanie wiedzy nie może zaczynać się zbyt wcześnie

# należy zaczynać od etapu ekspoloracji.

# Tylko długotrwałe próbkowanie jakiegoś zjawiska pozwala na racjonalne oszacowanie rozkładu danego zjawiska


"""
* Podana technika pozwala na dobre oszacowanie straty
"""
bandit_distribution_array
total_reward=0
number_trials
current_trial=0
while((number_trials-current_trial)>0):
    sample_array = sample(bandit_distribution_array)
    best_bandit = index_of(max(sample_array))
    reward =play(bandit_array[best_bandit])
    total_reward+=reward
    current_trial+=1
    update_distribution_array(best_bandit,reward)