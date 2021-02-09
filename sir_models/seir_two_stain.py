import numpy as np
import pandas as pd
import lmfit
from scipy.integrate import odeint
from lmfit import Parameters, minimize
from copy import deepcopy
from .utils import stepwise


def seir_step_two_stain(initial_conditions, t, params, history_store):
    sus_population = params['sus_population']
    r0_1 = params['r0_1']
    r0_2 = params['r0_2']
    delta = params['delta']
    gamma = params['gamma']
    alpha = params['alpha']
    rho = params['rho']

    S, E1, I1, E2, I2, R, D = initial_conditions

    q_coefs = {}
    for key, value in params.items():
        if key.startswith('t'):
            coef_t = int(key.split('_')[0][1:])
            q_coefs[coef_t] = value.value

    quarantine_mult = stepwise(t, q_coefs)
    rt_1 = (r0_1 - quarantine_mult * r0_1)# * S / sus_population
    beta_1 = rt_1 * gamma

    rt_2 = (r0_2 - quarantine_mult * r0_2)# * S / sus_population
    beta_2 = rt_2 * gamma


    new_exposed1 = beta_1 * I1 * (S / sus_population)
    new_infected1 = delta * E1
    new_dead1 = alpha * rho * I1
    new_recovered1 = gamma * (1 - alpha) * I1

    new_exposed2 = beta_2 * I2 * (S / sus_population)
    new_infected2 = delta * E2
    new_dead2 = alpha * rho * I2
    new_recovered2 = gamma * (1 - alpha) * I2

    dSdt = -new_exposed1 - new_exposed2

    dE1dt = new_exposed1 - new_infected1
    dI1dt = new_infected1 - new_recovered1 - new_dead1

    dE2dt = new_exposed2 - new_infected2
    dI2dt = new_infected2 - new_recovered2 - new_dead2

    dRdt = new_recovered1 + new_recovered2
    dDdt = new_dead1 + new_dead2

    assert sum([S, E1, I1, E2, I2, R, D]) - sus_population <= 1e10
    assert dSdt + dE1dt + dI1dt + dE2dt + dI2dt + dRdt + dDdt <= 1e10

    if history_store is not None:
        history_record = {
            't': t,
            'quarantine_mult': quarantine_mult,
            'rt_1': rt_1,
            'beta_1': beta_1,
            'rt_2': rt_2,
            'beta_2': beta_2,
            'new_exposed1': new_exposed1,
            'new_infected1': new_infected1,
            'new_dead1': new_dead1,
            'new_recovered1': new_recovered1,
            'new_exposed2': new_exposed2,
            'new_infected2': new_infected2,
            'new_dead2': new_dead2,
            'new_recovered2': new_recovered2,
        }
        history_store.append(history_record)

    return dSdt, dE1dt, dI1dt, dE2dt, dI2dt, dRdt, dDdt

# S -> E -> I -> R 
#             -> D

class SEIRTwoStain:
    def __init__(self, params=None):
        self.params = params

        self.train_data = None
        self.train_initial_conditions = None

        self.fit_result_ = None

    def get_fit_params(self, data):
        params = Parameters()
        params.add("base_population", value=12_000_000, vary=False)
        params.add("pre_existing_immunity", value=0.1806, vary=False)
        params.add("sus_population", expr='base_population - base_population * pre_existing_immunity', vary=False)
        params.add("r0_1", value=3.55, vary=False)
        params.add("new_stain_mult", value=1.5, vary=False)
        params.add("r0_2", expr='r0_1 * new_stain_mult', vary=False)

        params.add(f"t0_q", value=0.9, min=0, max=0.99, brute_step=0.1, vary=False)   

        # piece_size = 30
        # for t in range(piece_size, len(data), piece_size):
        #     params.add(f"t{t}_q", value=0.5, min=0, max=0.9, brute_step=0.1, vary=True)       


        params.add("delta", value=1/5.15, vary=False) # E -> I rate
        params.add("alpha", value=0.018, min=0.008, max=0.04, vary=False) # Probability to die if infected
        params.add("gamma", value=1/3.5, vary=False) # I -> R rate
        params.add("rho", value=1/14, vary=False) # I -> D rate
        return params


    def _predict(self, t, initial_conditions):
        history = []
        ret = odeint(seir_step_two_stain, initial_conditions, t, args=(self.params, history))
        history = pd.DataFrame(history)
        history.index = history.t
        return ret.T, history
