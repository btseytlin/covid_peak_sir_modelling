import numpy as np
import pandas as pd
import lmfit
from scipy.integrate import odeint
from lmfit import Parameters, minimize
from copy import deepcopy
from .utils import get_quarantine_multiplier_value


def seir_step(initial_conditions, t, params, history_store):
    sus_population = params['sus_population']
    r0 = params['r0']
    delta = params['delta']
    gamma = params['gamma']
    alpha = params['alpha']
    rho = params['rho']

    quarantine_mult = get_quarantine_multiplier_value(t, params)
    rt = r0 - quarantine_mult * r0 
    beta = rt * gamma

    S, E, I, R, D = initial_conditions

    new_exposed = beta * I * (S / sus_population)
    new_infected = delta * E
    new_dead = alpha * rho * I
    new_recovered = gamma * (1 - alpha) * I 

    dSdt = -new_exposed
    dEdt = new_exposed - new_infected
    dIdt = new_infected - new_recovered - new_dead
    dRdt = new_recovered
    dDdt = new_dead

    assert S + E + I + R + D - sus_population <= 1e10
    assert dSdt + dIdt + dEdt + dRdt + dDdt <= 1e10

    history_record = {
        't': t,
        'quarantine_mult': quarantine_mult,
        'rt': rt,
        'beta': beta,
        'new_exposed': new_exposed,
        'new_infected': new_infected,
        'new_dead': new_dead,
        'new_recovered': new_recovered,
    }
    history_store.append(history_record)

    return dSdt, dEdt, dIdt, dRdt, dDdt


def get_initial_coditions(model, data):
    # Simulate such initial params as to obtain as many deaths as in data

    sus_population = model.params['sus_population']
    alpha = model.params['alpha']
    rho = model.params['rho']
    delta = model.params['delta']


    old_params = deepcopy(model.params)
    for param_name, value in model.params.items():
        if param_name.startswith('t') and param_name.endswith('_q'):
            model.params[param_name].value = 0

    t = np.arange(365)
    (S, E, I, R, D), history = model._predict(t, (sus_population-1, 0, 1, 0, 0))
    fatality_day = np.argmax(D >= data.iloc[0].total_dead)
    I0 = I[fatality_day]
    E0 = E[fatality_day]
    Rec0 = R[fatality_day]
    D0 = D[fatality_day]
    S0 = S[fatality_day]

    model.params = old_params

    return (S0, E0, I0, Rec0, D0)


def residual(params, t, data, target, model_class, initial_conditions):
    model = model_class(params)

    (S, E, I, R, D), history = model._predict(t, initial_conditions)

    resid_D = (D - target['total_dead'])/target['total_dead']
    resid_I = (I.cumsum() - target['total_infected'])/target['total_infected']
    resid_R = (R - target['total_recovered'])/target['total_recovered']

    print(resid_D.mean(), resid_I.mean(), resid_R.mean())
    residuals = np.concatenate([
            resid_D,
            #resid_I,
            resid_R,
        ]).flatten()
    #print((residuals**2).sum())
    return residuals

# S -> E -> I -> R 
#             -> D

class SEIR:
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
        params.add("r0", value=3.55, vary=False)

        piece_size = 30
        for t in range(piece_size, len(data), piece_size):
          params.add(f"t{t}_q", value=0.5, min=0, max=0.9, brute_step=0.1, vary=True)       

        # params.add(f"t10_q", value=0, min=0.3, max=1.0, brute_step=0.1, vary=False)       
        
        # # Hard lockdown 05.05.20 - 31.05.20
        # params.add(f"t42_q", value=0.7, min=0.3, max=1.0, brute_step=0.1, vary=False)       

        # # Soft lockdown
        # params.add(f"t69_q", value=0.5, min=0.3, max=1.0, brute_step=0.1, vary=False)       

        # # Even softer lockdown
        # params.add(f"t165_q", value=0.4, min=0.3, max=1.0, brute_step=0.1, vary=False)       


        params.add("delta", value=1/5.15, vary=False) # E -> I rate
        params.add("alpha", value=0.018, min=0.008, max=0.04, vary=True) # Probability to die if infected
        params.add("gamma", value=1/3.5, vary=False) # I -> R rate
        params.add("rho", value=1/14, vary=False) # I -> D rate
        return params


    def fit(self, data):
        self.train_data = data

        y = data[['total_dead', 'total_infected', 'total_recovered']]

        self.params = self.get_fit_params(data)

        t = np.arange(len(data))
        initial_conditions = get_initial_coditions(self, self.train_data)
        self.train_initial_conditions = initial_conditions

        minimize_resut = minimize(residual, self.params, args=(t, data, y, SEIR, initial_conditions))

        self.fit_result_  = minimize_resut

        self.params = self.fit_result_.params
        return self


    def _predict(self, t, initial_conditions):
        history = []
        ret = odeint(seir_step, initial_conditions, t, args=(self.params, history))
        history = pd.DataFrame(history)
        history.index = history.t
        return ret.T, history

    def predict_train(self):
        train_data_steps = np.arange(len(self.train_data))
        return self._predict(train_data_steps, self.train_initial_conditions)

    def predict_test(self, t):
        (S, E, I, R, D), history = self.predict_train()

        test_initial_conditions = (S[-1], E[-1], I[-1], R[-1], D[-1])
        return self._predict(t, test_initial_conditions)
