import numpy as np
import pandas as pd
import lmfit
from scipy.integrate import odeint
from lmfit import Parameters, minimize
from copy import deepcopy
from tqdm.auto import tqdm
from .utils import stepwise


class DayAheadFitter:
    def __init__(self, result=None):
        self.result = result

    def fit(self, model, data, *args, **kwargs):
        params = model.get_fit_params(data)
        t = np.arange(len(data))

        minimize_resut = minimize(self.residual,
                                  params,
                                  *args,
                                  args=(t, data, model),
                                  **kwargs)

        self.result = minimize_resut
        model.params = self.result.params

    def residual(self, params, t_vals, data, model):
        model.params = params

        # print([params[p] for p in params if params[p].vary])
        eval_t = t_vals[1::10]
        true_D = np.zeros(len(eval_t))
        true_R = np.zeros(len(eval_t))
        preds_D = np.zeros(len(eval_t))
        preds_R = np.zeros(len(eval_t))
        for i, t in enumerate(eval_t):
            train_data = data.iloc[:t]
            initial_conditions = model.get_initial_conditions(train_data)
            # print(train_data.iloc[0])
            # print(initial_conditions)
            (S, E, I, R, D), history = model.predict([t-1, t], initial_conditions, history=False)
            true_D[i] = data.iloc[t].total_dead
            true_R[i] = data.iloc[t].total_recovered
            preds_D[i] = D[-1]
            preds_R[i] = R[-1]

            # print('True total dead', true_D[i])
            # print('Pred total dead', preds_D[i])
            # raise

        resid_D = (preds_D - true_D) #/(true_D+1e-6)

        residuals = np.concatenate([
            resid_D,
        ]).flatten()

        # print(residuals[:50])
        # print(residuals[-50:])
        model.maybe_log('Mae:', np.abs(residuals).mean())
        return residuals

# S -> E -> I -> R 
#             -> D


class BaseModel:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def maybe_log(self, *args):
        if self.verbose:
            print(*args)


class SEIR(BaseModel):
    def __init__(self, *args, params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params

    def step(self, initial_conditions, t, params, history_store):
        sus_population = params['sus_population']
        r0 = params['r0']
        delta = params['delta']
        gamma = params['gamma']
        alpha = params['alpha']
        rho = params['rho']

        q_coefs = {}
        for key, value in params.items():
            if key.startswith('t'):
                coef_t = int(key.split('_')[0][1:])
                q_coefs[coef_t] = value.value

        quarantine_mult = stepwise(t, q_coefs)
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

        if history_store is not None:
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

    def get_initial_conditions(self, data):
        # Simulate such initial params as to obtain as many deaths as in data

        sus_population = self.params['sus_population']

        t = np.arange(365)
        (S, E, I, R, D), history = self.predict(t, (sus_population - 1, 0, 1, 0, 0), history=False)
        fatality_day = np.argmax(D >= data.iloc[-1].total_dead)

        I0 = I[fatality_day]
        E0 = E[fatality_day]
        Rec0 = R[fatality_day]
        D0 = D[fatality_day]
        S0 = S[fatality_day]
        return (S0, E0, I0, Rec0, D0)

    def get_fit_params(self, data):
        params = Parameters()
        # Non-variable
        params.add("base_population", value=12_000_000, vary=False)
        params.add("pre_existing_immunity", value=0.1806, vary=False)
        params.add("sus_population", expr='base_population - base_population * pre_existing_immunity', vary=False)
        params.add("r0", value=3.55, vary=False)

        params.add("delta", value=1/5.15, vary=False) # E -> I rate
        params.add("alpha", value=0.0066, min=0.0066, max=0.05, vary=False) # Probability to die if infected
        params.add("gamma", value=1/3.5, vary=False) # I -> R rate
        params.add("rho", value=1/14, vary=False) # I -> D rate

        # Variable
        params.add(f"t0_q", value=0.5, min=0, max=0.99, brute_step=0.1, vary=True)
        piece_size = 60
        for t in range(piece_size, len(data), piece_size):
          params.add(f"t{t}_q", value=0.5, min=0, max=0.99, brute_step=0.1, vary=True)

        return params

    def predict(self, t, initial_conditions, history=True):
        if history:
            history = []
        else:
            history = None

        ret = odeint(self.step, initial_conditions, t, args=(self.params, history))

        if history:
            history = pd.DataFrame(history)
            if not history.empty:
                history.index = history.t
        return ret.T, history
