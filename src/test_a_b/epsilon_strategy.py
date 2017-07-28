"""
 * Strategia epsilon
 - epsilon value controls rate of exploring than using knowledge.
 - That all strategy is based on summing all winnings
 -
"""

epsilon = 0.1
best_bandit = 0# Indeks najlepszej maszyny w tablicy
bandit_array = 0 # Tablica obiektów reprezentujących maszyny
total_reward = 0
number_trials = 0
current_trial = 0

number_explore_trials = (1-epsilon)*number_trials

while((number_trials-current_trial)>0):
    if(current_trial < number_explore_trials):
        random_bandit = rand(0,len(bandit_array))
        total_reward += play(bandit_array[random_bandit])
        update_best_bandit()#update the best bandit
    else:
        total_reward +=play(bandit_array[best_bandit])
        current_trial+=1