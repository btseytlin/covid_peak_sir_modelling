import numpy as np
import pandas as pd
from copy import deepcopy
import lmfit
from scipy.integrate import odeint
from lmfit import Parameters
from .utils import stepwise_soft, shift, pad_left


class SEIR:
    def __init__(self, stepwise_size=60, params=None):
        super().__init__()
        self.stepwise_size = stepwise_size
        self.params = params

    def get_fit_params(self, data):
        params = Parameters()
        params.add("population", value=12_000_000, vary=False)

        params.add("sigmoid_r", value=20, min=1, max=30, brute_step=1, vary=False)
        params.add("sigmoid_c", value=0.5, min=0, max=1, brute_step=0.1, vary=False)

        params.add("epidemic_started_days_ago", value=10, min=1, max=90, brute_step=10, vary=False)

        params.add("r0", value=4, min=1.5, max=5, brute_step=0.05, vary=True)

        params.add("alpha", value=0.0064, min=0.005, max=0.0078, brute_step=0.0005, vary=False)  # IFR
        params.add("delta", value=1/3, min=1/14, max=1, vary=True)  # E -> I rate
        params.add("gamma", value=1/9, min=1/14, max=1/7, vary=False)  # I -> R rate
        params.add("rho", expr='gamma', vary=False)  # I -> D rate

        params.add("incubation_days", expr='1/delta', vary=False)
        params.add("infectious_days", expr='1/gamma', vary=False)

        params.add(f"t0_q", value=0, min=0, max=0.99, brute_step=0.1, vary=False)
        piece_size = self.stepwise_size
        for t in range(piece_size, len(data), piece_size):
            params.add(f"t{t}_q", value=0.5, min=0, max=0.99, brute_step=0.1, vary=True)
        return params

    def get_initial_conditions(self, data):
        # Simulate such initial params as to obtain as many deaths as in data
        population = self.params['population']
        epidemic_started_days_ago = self.params['epidemic_started_days_ago']

        new_params = deepcopy(self.params)
        for key, value in new_params.items():
            if key.startswith('t'):
                new_params[key].value = 0
        new_model = self.__class__(params=new_params)

        t = np.arange(epidemic_started_days_ago)
        (S, E, I, R, D), history = new_model.predict(t, (population - 1, 0, 1, 0, 0), history=False)

        I0 = I[-1]
        E0 = E[-1]
        Rec0 = R[-1]
        D0 = D[-1]
        S0 = S[-1]
        return (S0, E0, I0, Rec0, D0)

    def compute_daily_values(self, S, E, I, R, D):
        new_dead = pad_left(np.diff(D))
        new_recovered = pad_left(np.diff(R))
        new_infected = pad_left(np.diff(I)) + new_recovered + new_dead
        new_exposed = pad_left(np.diff(S[::-1])[::-1])

        return new_exposed, new_infected, new_recovered, new_dead

    def get_step_rt_beta(self, t, params):
        r0 = params['r0']
        gamma = params['gamma']
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
        return quarantine_mult, rt, beta

    def step(self, initial_conditions, t, params, history_store):
        population = params['population']
        delta = params['delta']
        gamma = params['gamma']
        alpha = params['alpha']
        rho = params['rho']

        quarantine_mult, rt, beta = self.get_step_rt_beta(t, params)

        S, E, I, R, D = initial_conditions

        new_exposed = beta * I * (S / population)
        new_infected = delta * E
        new_dead = alpha * rho * I
        new_recovered = gamma * (1 - alpha) * I

        dSdt = -new_exposed
        dEdt = new_exposed - new_infected
        dIdt = new_infected - new_recovered - new_dead
        dRdt = new_recovered
        dDdt = new_dead

        assert S + E + I + R + D - population <= 1e10
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

    def predict(self, t, initial_conditions, history=True):
        if history == True:
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


class SEIRHidden(SEIR):
    def get_fit_params(self, data):
        params = super().get_fit_params(data)

        params.add("pi", value=0.2, min=0.15, max=0.3, brute_step=0.01, vary=True)  # Probability to discover a new infected case in a day
        params.add("pd", value=0.35, min=0.15, max=0.9, brute_step=0.05, vary=True)  # Probability to discover a death
        return params

    def get_initial_conditions(self, data):
        population = self.params['population']
        epidemic_started_days_ago = self.params['epidemic_started_days_ago']

        new_params = deepcopy(self.params)
        for key, value in new_params.items():
            if key.startswith('t'):
                new_params[key].value = 0
        new_model = self.__class__(params=new_params)

        t = np.arange(epidemic_started_days_ago)
        (S, E, I, Iv, R, Rv, D, Dv), history = new_model.predict(t, (population-1, 0, 1, 0, 0, 0, 0, 0), history=False)

        S0 = S[-1]
        E0 = E[-1]
        I0 = I[-1]
        Iv0 = Iv[-1]
        R0 = R[-1]
        Rv0 = Rv[-1]
        D0 = D[-1]
        Dv0 = Dv[-1]
        return (S0, E0, I0, Iv0, R0, Rv0, D0, Dv0)

    def step(self, initial_conditions, t, params, history_store):
        population = params['population']
        delta = params['delta']
        gamma = params['gamma']
        alpha = params['alpha']
        rho = params['rho']
        pi = params['pi']
        pd = params['pd']

        quarantine_mult, rt, beta = self.get_step_rt_beta(t, params)

        (S, E, I, Iv, R, Rv, D, Dv) = initial_conditions

        new_exposed = beta * (I+Iv) * (S / population)
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

        assert S + E + I + Iv + R + Rv + D + Dv - population <= 1e10
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
        new_dead_invisible = pad_left(np.diff(D))
        new_recovered_invisible = pad_left(np.diff(R))
        new_recovered_visible = pad_left(np.diff(Rv))
        new_exposed = pad_left(np.diff(S[::-1])[::-1])

        new_dead_visible_from_Iv = self.params['alpha'] * self.params['rho'] * pad_left(shift(Iv, 1)[1:])
        new_dead_visible_from_I = pad_left(np.diff(Dv)) - new_dead_visible_from_Iv
        new_dead_visible = new_dead_visible_from_Iv + new_dead_visible_from_I

        new_infected_visible = pad_left(np.diff(Iv)) + new_recovered_visible + new_dead_visible_from_Iv
        new_infected_invisible = pad_left(np.diff(I)) + new_recovered_invisible + new_dead_visible_from_I

        return new_exposed, new_infected_invisible, new_infected_visible, new_recovered_invisible, new_recovered_visible, new_dead_invisible, new_dead_visible


class SEIRHiddenTwoStrains(SEIRHidden):

    @classmethod
    def from_strain_one_model(cls, model):
        strain1_params = model.params
        strain1_params.add("beta2_mult", value=1.5, min=1, max=2, vary=False)
        return cls(params=deepcopy(strain1_params))

    def get_fit_params(self, data):
        raise Exception('2 strain models can\'t be fit. Fit a one strain model, then use `from_strain_one_model`')

    def step(self, initial_conditions, t, params, history_store):
        population = params['population']
        delta = params['delta']
        gamma = params['gamma']
        alpha = params['alpha']
        rho = params['rho']
        pi = params['pi']
        pd = params['pd']
        beta2_mult = params['beta2_mult']

        quarantine_mult, rt1, beta1 = self.get_step_rt_beta(t, params)

        beta2 = beta2_mult * beta1
        (S, E1, I1, Iv1, E2, I2, Iv2, R, Rv, D, Dv) = initial_conditions

        new_exposed_strain1 = beta1 * (I1 + Iv1) * (S / population)
        new_infected_invisible_strain1 = (1 - pi) * delta * E1
        new_recovered_invisible_strain1 = gamma * (1 - alpha) * I1
        new_dead_invisible_strain1 = (1 - pd) * alpha * rho * I1
        new_dead_visible_from_I_strain1 = pd * alpha * rho * I1
        new_infected_visible_strain1 = pi * delta * E1
        new_recovered_visible_strain1 = gamma * (1 - alpha) * Iv1
        new_dead_visible_from_Iv_strain1 = alpha * rho * Iv1

        new_exposed_strain2 = beta2 * (I2 + Iv2) * (S / population)
        new_infected_invisible_strain2 = (1 - pi) * delta * E2
        new_recovered_invisible_strain2 = gamma * (1 - alpha) * I2
        new_dead_invisible_strain2 = (1 - pd) * alpha * rho * I2
        new_dead_visible_from_I_strain2 = pd * alpha * rho * I2
        new_infected_visible_strain2 = pi * delta * E2
        new_recovered_visible_strain2 = gamma * (1 - alpha) * Iv2
        new_dead_visible_from_Iv_strain2 = alpha * rho * Iv2

        dSdt = -(new_exposed_strain1 + new_exposed_strain2)

        dE1dt = new_exposed_strain1 - new_infected_visible_strain1 - new_infected_invisible_strain1
        dI1dt = new_infected_invisible_strain1 - new_recovered_invisible_strain1 - new_dead_invisible_strain1 - new_dead_visible_from_I_strain1
        dIv1dt = new_infected_visible_strain1 - new_recovered_visible_strain1 - new_dead_visible_from_Iv_strain1

        dE2dt = new_exposed_strain2 - new_infected_visible_strain2 - new_infected_invisible_strain2
        dI2dt = new_infected_invisible_strain2 - new_recovered_invisible_strain2 - new_dead_invisible_strain2 - new_dead_visible_from_I_strain2
        dIv2dt = new_infected_visible_strain2 - new_recovered_visible_strain2 - new_dead_visible_from_Iv_strain2

        dRvdt = new_recovered_visible_strain1 + new_recovered_visible_strain2
        dRdt = new_recovered_invisible_strain1 +  new_recovered_invisible_strain2
        dDdt = new_dead_invisible_strain1 + new_dead_invisible_strain2

        dDvdt = new_dead_visible_from_I_strain1 + new_dead_visible_from_Iv_strain1 + new_dead_visible_from_I_strain2 + new_dead_visible_from_Iv_strain2

        assert S + E1 + I1 + Iv1 + E2 + I2 + Iv2 + R + Rv + D + Dv - population <= 1e10
        assert dSdt + dE1dt + dI1dt + dIv1dt + dE2dt + dI2dt + dIv2dt + dRdt + dRvdt + dDdt + dDvdt <= 1e10

        if history_store is not None:
            history_record = {
                't': t,
                'quarantine_mult': quarantine_mult,
                'rt_strain1': rt1,
                'rt_strain2': beta2_mult * rt1,
                'beta_strain1': beta1,
                'beta_strain2': beta2,
                'new_exposed_strain1': new_exposed_strain1,
                'new_infected_visible_strain1': new_infected_visible_strain1,
                'new_dead_visible_strain1': new_dead_visible_from_I_strain1 + new_dead_visible_from_Iv_strain1,
                'new_recovered_visible_strain1': new_recovered_visible_strain1,
                'new_infected_invisible_strain1': new_infected_invisible_strain1,
                'new_dead_invisible_strain1': new_dead_invisible_strain1,
                'new_recovered_invisible_strain1': new_recovered_invisible_strain1,

                'new_exposed_strain2': new_exposed_strain2,
                'new_infected_visible_strain2': new_infected_visible_strain2,
                'new_dead_visible_strain2': new_dead_visible_from_I_strain2 + new_dead_visible_from_Iv_strain2,
                'new_recovered_visible_strain2': new_recovered_visible_strain2,
                'new_infected_invisible_strain2': new_infected_invisible_strain2,
                'new_dead_invisible_strain2': new_dead_invisible_strain2,
                'new_recovered_invisible_strain2': new_recovered_invisible_strain2,
            }
            history_store.append(history_record)

        return dSdt, dE1dt, dI1dt, dIv1dt,  dE2dt, dI2dt, dIv2dt, dRdt, dRvdt, dDdt, dDvdt

    def compute_daily_values(self, S, E1, I1, Iv1, E2, I2, Iv2, R, Rv, D, Dv):
        new_dead_invisible = pad_left(np.diff(D))
        new_recovered_invisible = pad_left(np.diff(R))
        new_recovered_visible = pad_left(np.diff(Rv))
        new_exposed = pad_left(np.diff(S[::-1])[::-1])

        new_dead_visible_from_Iv = self.params['alpha'] * self.params['rho'] * pad_left(shift(Iv1, 1)[1:])
        new_dead_visible_from_I = pad_left(np.diff(Dv)) - new_dead_visible_from_Iv
        new_dead_visible = new_dead_visible_from_Iv + new_dead_visible_from_I

        new_infected_visible = pad_left(np.diff(Iv1)) + new_recovered_visible + new_dead_visible_from_Iv
        new_infected_invisible = pad_left(np.diff(I1)) + new_recovered_invisible + new_dead_visible_from_I

        return new_exposed, new_infected_invisible, new_infected_visible, new_recovered_invisible, new_recovered_visible, new_dead_invisible, new_dead_visible
