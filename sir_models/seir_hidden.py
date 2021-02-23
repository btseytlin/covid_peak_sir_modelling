import numpy as np
import pandas as pd
import lmfit
from scipy.integrate import odeint
from lmfit import Parameters, minimize
from copy import deepcopy
from tqdm.auto import tqdm
from .utils import stepwise_soft
from .seir import SEIR, BaseFitter


class HiddenCurveFitter(BaseFitter):
    def get_initial_conditions(self, model, data):
        # Simulate such initial params as to obtain as many deaths as in data

        sus_population = model.params['sus_population']

        new_params = deepcopy(model.params)
        for key, value in new_params.items():
            if key.startswith('t'):
                new_params[key].value = 0
        new_model = SEIRHidden(params=new_params)

        t = np.arange(365)
        (S, E, I, Iv, R, Rv, D, Dv), history = new_model.predict(t, (sus_population-1, 0, 1, 0, 0, 0, 0, 0), history=False)
        fatality_day = np.argmax(Dv >= data.iloc[0].total_dead)

        S0 = S[fatality_day]
        E0 = E[fatality_day]
        I0 = I[fatality_day]
        Iv0 = Iv[fatality_day]
        R0 = R[fatality_day]
        Rv0 = Rv[fatality_day]
        D0 = D[fatality_day]
        Dv0 = Dv[fatality_day]
        return (S0, E0, I0, Iv0, R0, Rv0, D0, Dv0)

    def residual(self, params, t_vals, data, model):
        model.params = params

        initial_conditions = self.get_initial_conditions(model, data)
        (S, E, I, Iv, R, Rv, D, Dv), history = model.predict(t_vals, initial_conditions, history=False)

        resids = []

        if self.use_dead:
            resid_D = (Dv - data['total_dead']) / data['total_dead'].std()
            resids.append(resid_D)
        if self.use_recovered:
            resid_R = (Rv - data['total_recovered']) / data['total_recovered'].std() / 30
            resids.append(resid_R)

        residuals = np.concatenate(resids).flatten()
        return residuals



# S -> E -> I -> R 
#             -> D


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
