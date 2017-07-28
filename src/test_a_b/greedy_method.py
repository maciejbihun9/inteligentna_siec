epsilon=0.1
best_bandit
bandit_array
total_reward=0
number_trials
current_trial=0

from numpy import *

while((number_trials-current_trial)>0):
    random_float = rand(0,1)
    # eksploracja przestrzeni rowiązań
    if(random_float<epsilon):
        random_bandit = rand(0,len(bandit_array))
        total_reward += play(bandit_array[random_bandit])
        update_best_bandit()
    else:
        total_reward +=play(bandit_array[best_bandit])
        current_trial+=1