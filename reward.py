def compute_reward(score):
    if score >= 1.5:
        return 2
    elif score >= 1:
        return 1
    else:
        return -1
