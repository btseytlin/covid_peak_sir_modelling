import numpy as np
import pandas as pd
import lmfit
from scipy.integrate import odeint
from lmfit import Parameters, minimize
from copy import deepcopy
from tqdm.auto import tqdm
from scipy.special import softmax
from .utils import stepwise_soft, shift
from .seir import SEIR, CurveFitter


class HiddenCurveFitter(CurveFitter):
    def get_initial_conditions(self, model, data):
        sus_population = model.params['sus_population']
        epidemic_started_days_ago = model.params['epidemic_started_days_ago']

        new_params = deepcopy(model.params)
        for key, value in new_params.items():
            if key.startswith('t'):
                new_params[key].value = 0
        new_model = SEIRHidden(params=new_params)

        t = np.arange(epidemic_started_days_ago)
        (S, E, I, Iv, R, Rv, D, Dv), history = new_model.predict(t, (sus_population-1, 0, 1, 0, 0, 0, 0, 0), history=False)

        S0 = S[-1]
        E0 = E[-1]
        I0 = I[-1]
        Iv0 = Iv[-1]
        R0 = R[-1]
        Rv0 = Rv[-1]
        D0 = D[-1]
        Dv0 = Dv[-1]
        return (S0, E0, I0, Iv0, R0, Rv0, D0, Dv0)

    def residual(self, params, t_vals, data, model):
        model.params = params

        initial_conditions = self.get_initial_conditions(model, data)

        (S, E, I, Iv, R, Rv, D, Dv), history = model.predict(t_vals, initial_conditions, history=True)
        (new_exposed,
         new_infected_invisible, new_infected_visible,
         new_recovered_invisible,
         new_recovered_visible,
         new_dead_invisible, new_dead_visible) = model.compute_daily_values(S, E, I, Iv, R, Rv, D, Dv)
        true_daily_cases = data[self.new_cases_col][:len(new_infected_visible)].fillna(0)
        true_daily_deaths = data[self.new_deaths_col][:len(new_dead_visible)].fillna(0)

        resid_I_new = (new_infected_visible - true_daily_cases) / (np.maximum(new_infected_visible, true_daily_cases) + 1e-10)

        resid_D_new = (new_dead_visible - true_daily_deaths) / (np.maximum(new_dead_visible, true_daily_deaths) + 1e-10)

        residuals = np.concatenate([
            resid_I_new,
            resid_D_new,
        ]).flatten()
        return residuals


class EnsembleFitter(HiddenCurveFitter):
    def optimize(self, params, t, data, model, args, kwargs):
        params_history = []
        error_history = []

        with tqdm(total=self.max_iters) as pbar:
            def callback(params, iter, resid, *args, **kwargs):
                if iter % 10 == 0:
                    pbar.n = iter
                    pbar.refresh()
                    error = np.abs(resid).mean()
                    pbar.set_postfix({"MARE": error})

                    params_history.append(params)
                    error_history.append(error)

            minimize_resut = minimize(self.residual,
                                      params,
                                      *args,
                                      args=(t, data, model),
                                      iter_cb=callback,
                                      max_nfev=self.max_iters,
                                      **kwargs)

        weights = softmax(error_history)
        aggregate_params = deepcopy(params_history[-1])
        for param_name in params.keys():
            agg_value = 0
            for i in range(len(params_history)):
                agg_value += weights[i] * params_history[i][param_name].value
            aggregate_params[param_name].value = agg_value
        minimize_resut.params = aggregate_params

        return minimize_resut


class SEIRHidden(SEIR):
    def step(self, initial_conditions, t, params, history_store):
        sus_population = params['sus_population']
        r0 = params['r0']
        delta = params['delta']
        gamma = params['gamma']
        alpha = params['alpha']
        rho = params['rho']
        pi = params['pi']
        pd = params['pd']

        q_coefs = {}
        for key, value in params.items():
            if key.startswith('t'):
                coef_t = int(key.split('_')[0][1:])
                q_coefs[coef_t] = value.value

        quarantine_mult = stepwise_soft(t, q_coefs)
        rt = r0 - quarantine_mult * r0
        beta = rt * gamma

        (S, E, I, Iv, R, Rv, D, Dv) = initial_conditions

        new_exposed = beta * I * (S / sus_population)
        new_infected_invisible = (1 - pi) * delta * E
        new_recovered_invisible = gamma * (1 - alpha) * I
        new_dead_invisible = (1 - pd) * alpha * rho * I
        new_dead_visible_from_I = pd * alpha * rho * I

        new_infected_visible = pi * delta * E
        new_recovered_visible = gamma * (1 - alpha) * Iv
        new_dead_visible_from_Iv = alpha * rho * Iv

        dSdt = -new_exposed
        dEdt = new_exposed - new_infected_visible - new_infected_invisible
        dIdt = new_infected_invisible - new_recovered_invisible - new_dead_invisible - new_dead_visible_from_I
        dIvdt = new_infected_visible - new_recovered_visible - new_dead_visible_from_Iv
        dRdt = new_recovered_invisible
        dRvdt = new_recovered_visible
        dDdt = new_dead_invisible
        dDvdt = new_dead_visible_from_I + new_dead_visible_from_Iv

        assert S + E + I + Iv + R + Rv + D + Dv - sus_population <= 1e10
        assert dSdt + dEdt + dIdt + dIvdt + dRdt + dRvdt + dDdt + dDvdt <= 1e10

        if history_store is not None:
            history_record = {
                't': t,
                'quarantine_mult': quarantine_mult,
                'rt': rt,
                'beta': beta,
                'new_exposed': new_exposed,
                'new_infected_visible': new_infected_visible,
                'new_dead_visible': new_dead_visible_from_I + new_dead_visible_from_Iv,
                'new_recovered_visible': new_recovered_visible,
                'new_infected_invisible': new_infected_invisible,
                'new_dead_invisible': new_dead_invisible,
                'new_recovered_invisible': new_recovered_invisible,
            }
            history_store.append(history_record)

        return dSdt, dEdt, dIdt, dIvdt, dRdt, dRvdt, dDdt, dDvdt

    def compute_daily_values(self, S, E, I, Iv, R, Rv, D, Dv):
        new_dead_invisible = np.diff(D)
        new_recovered_invisible = np.diff(R)
        new_recovered_visible = np.diff(Rv)
        new_exposed = np.diff(S[::-1])[::-1]

        new_dead_visible_from_Iv = self.params['alpha'] * self.params['rho'] * shift(Iv, 1)[1:]
        new_dead_visible_from_I = np.diff(Dv) - new_dead_visible_from_Iv
        new_dead_visible = new_dead_visible_from_Iv + new_dead_visible_from_I

        new_infected_visible = np.diff(Iv) + new_recovered_visible + new_dead_visible_from_Iv
        new_infected_invisible = np.diff(I) + new_recovered_invisible + new_dead_visible_from_I

        return new_exposed, new_infected_invisible, new_infected_visible, new_recovered_invisible, new_recovered_visible, new_dead_invisible, new_dead_visible

    def get_fit_params(self, data):
        params = super().get_fit_params(data)

        params.add("pi", value=0.5, min=0.01, max=1, vary=True)  # Probability to discover a new infected case in a day
        params.add("pd", value=0.5, min=0.01, max=1, vary=True)  # Probability to discover a death
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
