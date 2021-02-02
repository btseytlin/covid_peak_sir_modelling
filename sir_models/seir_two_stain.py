import numpy as np
import pandas as pd
import lmfit
from scipy.integrate import odeint
from lmfit import Parameters, minimize
from copy import deepcopy
from .utils import get_quarantine_multiplier_value


def seir_step_two_stain(initial_conditions, t, params, history_store):
    sus_population = params['sus_population']
    r0_1 = params['r0_1']
    r0_2 = params['r0_2']
    delta = params['delta']
    gamma = params['gamma']
    alpha = params['alpha']
    rho = params['rho']

    quarantine_mult = get_quarantine_multiplier_value(t, params)
    rt_1 = r0_1 - quarantine_mult * r0_1
    beta_1 = rt_1 * gamma

    rt_2 = r0_2 - quarantine_mult * r0_2
    beta_2 = rt_2 * gamma

    S, E1, I1, E2, I2, R, D = initial_conditions

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

    t = np.arange(400)
    (S, E1, I1, E2, I2, R, D), history = model._predict(t, (sus_population-1, 0, 1, 0, 0, 0, 0))

    fatality_day = np.argmax(D >= data.iloc[0].total_dead)
    I1_0 = I1[fatality_day]
    E1_0 = E1[fatality_day]

    I2_0 = 0
    E2_0 = 0

    Rec_0 = R[fatality_day]
    D_0 = D[fatality_day]
    S_0 = S[fatality_day]

    model.params = old_params

    return (S_0, E1_0, I1_0, E2_0, I2_0, Rec_0, D_0)


def residual(params, t, data, target, model_class, initial_conditions):
    model = model_class(params)

    (S, E1, I1, E2, I2, R, D), history = model._predict(t, initial_conditions)

    resid_D = D - target[:, 0]
    resid_I = (I1 + I2).cumsum() - target[:, 1]

    # print(resid_D.sum(), 1e-3*resid_I.sum())
    residuals = np.concatenate([
            resid_D,
            #1e-3*resid_I,
        ]).flatten()
    #print((residuals**2).sum())
    return residuals

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

        #params.add("new_stain_ratio", value=0.001, min=0, max=1, vary=False)

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
        params.add("alpha", value=0.018, min=0.008, max=0.04, vary=False) # Probability to die if infected
        params.add("gamma", value=1/3.5, vary=False) # I -> R rate
        params.add("rho", value=1/14, vary=False) # I -> D rate
        return params


    def fit(self, data):
        self.train_data = data

        y = data[['total_dead', 'total_infected']].values

        self.params = self.get_fit_params(data)

        t = np.arange(len(data))
        initial_conditions = get_initial_coditions(self, self.train_data)
        self.train_initial_conditions = initial_conditions

        minimize_resut = minimize(residual, self.params, args=(t, data, y, SEIRTwoStain, initial_conditions))

        self.fit_result_  = minimize_resut

        self.params = self.fit_result_.params
        return self


    def _predict(self, t, initial_conditions):
        history = []
        ret = odeint(seir_step_two_stain, initial_conditions, t, args=(self.params, history))
        history = pd.DataFrame(history)
        history.index = history.t
        return ret.T, history

    def predict_train(self):
        train_data_steps = np.arange(len(self.train_data))
        return self._predict(train_data_steps, self.train_initial_conditions)

    def predict_test(self, t):
        (S, E1, I1, E2, I2, R, D), history = self.predict_train()

        test_initial_conditions = (S[-1], E1[-1], I1[-1], 0, 10, R[-1], D[-1])
        return self._predict(t, test_initial_conditions)
