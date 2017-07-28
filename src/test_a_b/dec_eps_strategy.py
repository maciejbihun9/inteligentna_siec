epsilon=1
best_bandit = 0
bandit_array = 0
total_reward= 0
number_trials = 0
current_trial=0
while((number_trials-current_trial)>0):
    random_float = rand(0,1)
    if(random_float<epsilon):
        random_bandit = rand(0,len(bandit_array))
        total_reward += play(bandit_array[random_bandit])
        update_best_bandit()
    else:
        total_reward += play(bandit_array[best_bandit])
    current_trial+=1
    epsilon = update_epsilon(epsilon)