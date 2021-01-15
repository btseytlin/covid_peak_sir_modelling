import numpy as np
import lmfit
from scipy.integrate import odeint


def sir_step_one_stain(initial_conditions, t, population, beta, gamma):
    S, I, R = initial_conditions
    
    new_infected = beta * I * (S / population)
    new_recovered = gamma * I 
    
    dSdt = -new_infected
    dIdt = new_infected - new_recovered
    dRdt = new_recovered
    return dSdt, dIdt, dRdt


def seir_step_one_stain(initial_conditions, t, population, beta, gamma, delta):
    S, E, I, R = initial_conditions
    
    new_exposed = beta * I * (S / population) - delta * E
    new_infected = delta * E
    new_recovered = gamma * I 
    
    dSdt = -new_exposed
    dEdt = new_exposed - new_infected
    dIdt = new_infected - new_recovered
    dRdt = new_recovered
    return dSdt, dEdt, dIdt, dRdt


class BaseModel:
    def __init__(self):
        pass

    def step(self, initial_conditions, *args):
        pass

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass


class SIROneStain(BaseModel):
    def __init__(self, population, beta=1.2, gamma=0.009):
        self.population = population
        self.beta = beta
        self.gamma = gamma

        self.fit_result_ = None

    def get_initial_coditions(self, infected, total_recovered):
        I0 = infected
        Rec0 = total_recovered
        S0 = self.population - I0 - Rec0
        return (S0, I0, Rec0)


    def fit(self, data, y):

        param_hints = {"beta": (1.2, 0.5, 2), 
                        "gamma": (0.009, 0, 1)} 

        def fitter(x, beta, gamma):
            initial_conditions = self.get_initial_coditions(data.iloc[x[0]].infected, data.iloc[x[0]].total_recovered)
            S, I, R = SIROneStain(self.population, beta, gamma)._predict(x, initial_conditions)
            return I

        mod = lmfit.Model(fitter)

        for kwarg, (init, mini, maxi) in param_hints.items():
            mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)

        params = mod.make_params()

        t = np.arange(len(data))

        self.fit_result_  = mod.fit(y, params, method="least_squares", x=t)

        best_params = self.fit_result_ .best_values
        self.beta = best_params['beta']
        self.gamma = best_params['gamma']
        return self


    def _predict(self, t, initial_conditions):
        ret = odeint(sir_step_one_stain, initial_conditions, t, 
            args=(self.population, self.beta, self.gamma))
        S, I, R = ret.T
        return S, I, R

    def predict(self, data, steps=None):
        initial_conditions = self.get_initial_coditions(data.iloc[0].infected, data.iloc[0].total_recovered)
        if steps is None:
            steps = len(data)

        t = np.arange(steps)
        return self._predict(t, initial_conditions)



class SEIROneStain(BaseModel):
    def __init__(self, population, beta=None, gamma=None, delta=None):
        self.population = population
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def fit(self, data, y):

        param_hints = {"beta": (1.2, 0.5, 3), 
               "gamma": (0.009, 0, 1),
               "delta": (1/9, 0, 1)}  

        def fitter(x, beta, gamma, delta):
            # Initial conditions
            E0 = data.iloc[x[0]].infected - delta * data.iloc[x[0]].infected
            I0 = delta * data.iloc[x[0]].infected
            Rec0 = data.iloc[x[0]].total_recovered
            S0 = self.population - I0 - E0 - Rec0
            initial_conditions = (S0, E0, I0, Rec0)
            S, E, I, R = SEIROneStain(self.population, beta, gamma, delta).predict(x, initial_conditions)
            return I

        mod = lmfit.Model(fitter)

        for kwarg, (init, mini, maxi) in param_hints.items():
            mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)

        params = mod.make_params()

        t = np.arange(len(data))
        self.fit_result_  = mod.fit(y, params, method="least_squares", x=t)

        best_params = self.fit_result_.best_values
        self.beta = best_params['beta']
        self.gamma = best_params['gamma']
        self.delta = best_params['delta']
        return self

    def predict(self, t, initial_conditions):
        ret = odeint(seir_step_one_stain, initial_conditions, t, 
            args=(self.population, self.beta, self.gamma, self.delta))
        S, E, I, R = ret.T
        return S, E, I, R



