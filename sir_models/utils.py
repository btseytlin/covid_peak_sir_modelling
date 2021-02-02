import numpy as np


def stepwise(t, coefficients):    
    t_arr = np.array(list(coefficients.keys()))

    min_index = np.min(t_arr)
    max_index = np.max(t_arr)

    if t <= min_index:
        index = min_index
    elif t >= max_index:
        index = max_index
    else:
        index = np.min(t_arr[t_arr >= t])
    return coefficients[index]

def get_quarantine_multiplier_value(t, params):
    q_coefs = {}
    for key, value in params.items():
        if key.startswith('t'):
            coef_t = int(key.split('_')[0][1:])
            q_coefs[coef_t] = value.value

    quarantine_mult = stepwise(t, q_coefs)
    return quarantine_mult
