import numpy as np
import pandas as pd
import lmfit
from scipy.integrate import odeint
from lmfit import Parameters, minimize
from copy import deepcopy
from tqdm.auto import tqdm
from .utils import stepwise_soft


class BaseFitter:
    def __init__(self,
                 result=None,
                 verbose=True,
                 max_iters=None,
                 brute_params=None,
                 save_params_every=100):
        self.result = result
        self.verbose = verbose
        self.max_iters = max_iters
        self.brute_params = brute_params or []

        self.save_params_every = save_params_every
        self.params_history = []
        self.error_history = []

    def optimize(self, params, t, data, model, args, kwargs):
        with tqdm(total=self.max_iters) as pbar:
            def callback(params, iter, resid, *args, **kwargs):
                if iter % 10 == 0:
                    pbar.n = iter
                    pbar.refresh()
                    pbar.set_postfix({"MARE": np.abs(resid).mean()})

                if iter % self.save_params_every == 0 and iter > 0:
                    error = np.abs(resid).mean()
                    self.params_history.append(deepcopy(params))
                    self.error_history.append(error)

            minimize_resut = minimize(self.residual,
                                      params,
                                      *args,
                                      args=(t, data, model),
                                      iter_cb=callback,
                                      max_nfev=self.max_iters,
                                      **kwargs)
        return minimize_resut

    def optimize_brute(self, params, param_name, brute_params, t, data, model, args, kwargs):
        best_result = None
        param = params[param_name]
        assert not param.vary
        param_min, param_max, param_step = param.min, param.max, param.brute_step

        last_params = deepcopy(params)
        iterator = tqdm(range(param_min, param_max + 1, param_step))
        for param_val in iterator:
            iterator.set_postfix({param_name: param_val})

            temp_params = last_params
            temp_params[param_name].value = param_val
            if brute_params:
                result = self.optimize_brute(temp_params, brute_params[0], brute_params[1:], t, data, model, args, kwargs)
            else:
                result = self.optimize(temp_params, t, data, model, args=args, kwargs=kwargs)

            if not best_result or np.mean(np.abs(result.residual)) < np.mean(np.abs(best_result.residual)):
                best_result = result

            # Start optimization from last best point
            last_params = deepcopy(result.params)

        return best_result

    def fit(self, model, data, *args, **kwargs):
        params = model.get_fit_params(data)
        t = np.arange(len(data))

        if not self.brute_params:
            self.result = self.optimize(params, t, data, model, args=args, kwargs=kwargs)
        else:
            self.result = self.optimize_brute(params, self.brute_params[0], self.brute_params[1:], t, data, model, args, kwargs)

        model.params = self.result.params


class CurveFitter(BaseFitter):
    def __init__(self, *args,
                 total_deaths_col='total_deaths',
                 new_deaths_col='new_deaths',
                 total_cases_col='total_cases',
                 new_cases_col='new_cases',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.new_deaths_col = new_deaths_col
        self.total_deaths_col = total_deaths_col

        self.total_cases_col = total_cases_col
        self.new_cases_col = new_cases_col

    def get_initial_conditions(self, model, data):
        # Simulate such initial params as to obtain as many deaths as in data
        sus_population = model.params['sus_population']
        epidemic_started_days_ago = model.params['epidemic_started_days_ago']

        new_params = deepcopy(model.params)
        for key, value in new_params.items():
            if key.startswith('t'):
                new_params[key].value = 0
        new_model = SEIR(params=new_params)

        t = np.arange(epidemic_started_days_ago)
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

        (S, E, I, R, D), history = model.predict(t_vals, initial_conditions, history=True)
        new_exposed, new_infected, new_recovered, new_dead = model.compute_daily_values(S, E, I, R, D)
        true_daily_cases = data[self.new_cases_col][:len(new_infected)].fillna(0)
        true_daily_deaths = data[self.new_deaths_col][:len(new_dead)].fillna(0)

        resid_I_new = (new_infected - true_daily_cases) / (np.maximum(new_infected, true_daily_cases) + 1e-10)

        resid_D_new = (new_dead - true_daily_deaths) / (np.maximum(new_dead, true_daily_deaths) + 1e-10)

        residuals = np.concatenate([
            resid_I_new,
            resid_D_new,
        ]).flatten()
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

    def compute_daily_values(self, S, E, I, R, D):
        new_dead = np.diff(D)
        new_recovered = np.diff(R)
        new_infected = np.diff(I) + new_recovered + new_dead
        new_exposed = np.diff(S[::-1])[::-1]

        return new_exposed, new_infected, new_recovered, new_dead

    def step(self, initial_conditions, t, params, history_store):
        sus_population = params['sus_population']
        r0 = params['r0']
        delta = params['delta']
        gamma = params['gamma']
        alpha = params['alpha']
        rho = params['rho']

        sigmoid_r = params['sigmoid_r']
        sigmoid_c = params['sigmoid_c']

        q_coefs = {}
        for key, value in params.items():
            if key.startswith('t'):
                coef_t = int(key.split('_')[0][1:])
                q_coefs[coef_t] = value.value

        quarantine_mult = stepwise_soft(t, q_coefs, r=sigmoid_r, c=sigmoid_c)
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

        params.add("rho", value=1 / 14, min=1 / 20, max=1 / 7, vary=False)  # I -> D rate
        params.add("alpha", value=0.0066, min=0.0001, max=0.05, vary=False)  # Probability to die if infected

        params.add("sigmoid_r", value=20, min=1, max=30, vary=False)
        params.add("sigmoid_c", value=0.5, min=0, max=1, vary=False)
        params.add("epidemic_started_days_ago", value=10, min=1, max=90, brute_step=10, vary=False)

        params.add(f"t0_q", value=0, min=0, max=0.99, brute_step=0.1, vary=False)
        # Variable
        params.add("r0", value=3.5, min=2.5, max=4, vary=True)
        params.add("delta", value=1 / 5.15, min=1 / 8, max=1 / 3, vary=True)  # E -> I rate
        params.add("gamma", value=1 / 9.5, min=1 / 20, max=1 / 2, vary=True)  # I -> R rate
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
                history = history[~history.index.duplicated(keep='first')]
        return ret.T, history
