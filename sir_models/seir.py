import numpy as np
import pandas as pd
import lmfit
from scipy.integrate import odeint
from lmfit import Parameters, minimize
from copy import deepcopy
from tqdm.auto import tqdm
from .utils import stepwise, stepwise_soft


class BaseFitter:
    def __init__(self,
                 use_dead=True,
                 use_recovered=False,
                 result=None,
                 verbose=True,
                 max_iters=None):
        self.use_dead = use_dead
        self.use_recovered = use_recovered
        self.result = result
        self.verbose = verbose
        self.max_iters = max_iters

    def fit(self, model, data, *args, **kwargs):
        params = model.get_fit_params(data)
        t = np.arange(len(data))

        callback = None
        if self.verbose:
            def callback(params, iter, resid, *args, **kwargs):
                if iter % 10 == 0:
                    print(f'Iter {iter} | MAE: {np.abs(resid).mean():0.4f}')

        minimize_resut = minimize(self.residual,
                                  params,
                                  *args,
                                  args=(t, data, model),
                                  iter_cb = callback,
                                  max_nfev = self.max_iters,
                                  **kwargs)

        self.result = minimize_resut
        model.params = self.result.params


class DayAheadFitter(BaseFitter):
    def __init__(self, *args, n_eval_points=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_eval_points = n_eval_points

    def get_initial_conditions(self, model, data):
        # Simulate such initial params as to obtain as many deaths as in data

        sus_population = model.params['sus_population']

        t = np.arange(365)
        (S, E, I, R, D), history = model.predict(t, (sus_population - 1, 0, 1, 0, 0), history=False)
        fatality_day = np.argmax(D >= data.iloc[-1].total_dead) # Last day!

        I0 = I[fatality_day]
        E0 = E[fatality_day]
        Rec0 = R[fatality_day]
        D0 = D[fatality_day]
        S0 = S[fatality_day]
        return (S0, E0, I0, Rec0, D0)

    def residual(self, params, t_vals, data, model):
        model.params = params

        eval_every = 1
        if self.n_eval_points:
            eval_every = len(data) // self.n_eval_points
        eval_t = t_vals[1::eval_every]
        resid_D = []
        resid_R = []

        iterator = enumerate(eval_t)
        if self.verbose:
            iterator = tqdm(iterator, total=len(eval_t))

        for i, t in iterator:
            train_data = data.iloc[:t]
            initial_conditions = self.get_initial_conditions(model, train_data)
            (S, E, I, R, D), history = model.predict([t-1, t], initial_conditions, history=False)

            if self.use_dead:
                resid_D.append((D[-1] - data.iloc[t].total_dead))
            if self.use_recovered:
                resid_R.append((R[-1] - data.iloc[t].total_recovered))

        resids = []
        if self.use_dead:
            resids.append(resid_D)
        if self.use_recovered:
            resids.append(resid_R)

        residuals = np.concatenate(resids).flatten()

        return residuals


class CurveFitter(BaseFitter):
    def get_initial_conditions(self, model, data):
        # Simulate such initial params as to obtain as many deaths as in data
        sus_population = model.params['sus_population']

        first_case_offset = model.params['first_case_offset'].value
        first_data_point_date = '2020-03-12'
        first_case_date = pd.to_datetime(first_data_point_date) - pd.to_timedelta(first_case_offset, 'days')
        days_since_first_case = (data.iloc[0].date - first_case_date).days

        new_params = deepcopy(model.params)
        for key, value in new_params.items():
            if key.startswith('t'):
                new_params[key].value = 0
        new_model = SEIR(params=new_params)

        t = np.arange(days_since_first_case)
        (S, E, I, R, D), history = new_model.predict(t, (sus_population - 1, 0, 1, 0, 0), history=False)

        I0 = I[-1]
        E0 = E[-1]
        Rec0 = R[-1]
        D0 = D[-1]
        S0 = S[-1]
        return (S0, E0, I0, Rec0, D0)

    def residual(self, params, t_vals, data, model):
        model.params = params

        initial_conditions = self.get_initial_conditions(model, data)

        (S, E, I, R, D), history = model.predict(t_vals, initial_conditions, history=False)

        resids = []
        if self.use_dead:
            resid_D = (D - data['total_dead'])
            resids.append(resid_D)
        if self.use_recovered:
            resid_R = (R - data['total_recovered'])
            resids.append(resid_R)

        residuals = np.concatenate(resids).flatten()
        return residuals


# S -> E -> I -> R 
#             -> D


class BaseModel:
    pass


class SEIR(BaseModel):
    def __init__(self, stepwise_size=30, params=None):
        super().__init__()
        self.stepwise_size = stepwise_size
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

    def get_fit_params(self, data):
        params = Parameters()
        # Non-variable
        params.add("base_population", value=12_000_000, vary=False)
        params.add("pre_existing_immunity", value=0.1806, vary=False)
        params.add("sus_population", expr='base_population - base_population * pre_existing_immunity', vary=False)
        params.add("r0", value=3.55, vary=False)

        params.add("delta", value=1/5.15, vary=False) # E -> I rate
        params.add("gamma", value=1/3.5, vary=False) # I -> R rate
        params.add("rho", value=1/14, vary=False) # I -> D rate

        # Variable
        # First data point: 2020-03-12
        # Calculate first case date as (2020-03-12 - first_case_offset days)
        params.add("first_case_offset", value=30, min=0, max=90, vary=True)

        params.add("alpha", value=0.0066, min=0.0001, max=0.05, vary=True) # Probability to die if infected

        params.add(f"t0_q", value=0, min=0, max=0.99, brute_step=0.1, vary=True)
        piece_size = self.stepwise_size
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
