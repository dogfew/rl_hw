def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """
    res = 0 
    for next_state, next_state_value in state_values.items():
        transition_prob = mdp.get_transition_prob(state, action, next_state)
        res += transition_prob * (mdp.get_reward(state, action, next_state) + gamma * next_state_value)
    return res
