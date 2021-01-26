import numpy as np
import lmfit
from scipy.integrate import odeint
from lmfit import Parameters, minimize


def seir_step(initial_conditions, t, params):
    population, beta, delta, gamma, alpha, rho = params
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
    return dSdt, dEdt, dIdt, dRdt, dDdt


# R = gamma * (1 - alpha) * I

# D = alpha * rho * I
# I = D / alpha / rho

def get_initial_coditions(model, data):
    D0 = data.total_dead.iloc[0]
    I0 = D0 / model.alpha / model.rho
    E0 = 0
    Rec0 = data.total_recovered.iloc[0]
    S0 = model.population - I0 - Rec0 - E0
    return (S0, E0, I0, Rec0, D0)


def residual(params, t, data, target, model_class, i0):
    model = model_class(population=params['population'],
                        beta=params['beta'],
                        delta=params['delta'],
                        gamma=params['gamma'],
                        alpha=params['alpha'],
                        rho=params['rho'],
                        i0=i0)

    initial_conditions = get_initial_coditions(model, data)

    S, E, I, R, D = model._predict(t, initial_conditions)

    residuals = np.concatenate([
            D - target[:, 0],
            (I.cumsum() - target[:, 1]),
        ]).flatten()

    return residuals

# S -> E -> I -> R 
#             -> D

class SEIR:
    def __init__(self, population, 
                        beta=None, # S -> E rate
                        delta=None, # E -> I rate
                        gamma=None, # I -> R rate
                        alpha=None, # I -> D rate
                        rho=None, # I -> D rate,
                        i0=None,
                ):
        self.population = population
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.alpha = alpha
        self.rho = rho
        self.i0 = i0 

        self.fit_result_ = None

        self.train_data = None

    @property
    def params(self):
        return (self.population, self.beta, self.delta, self.gamma, self.alpha, self.rho)

    def get_fit_params(self):
        params = Parameters()
        params.add("population", value=self.population, vary=False)
        params.add("beta", value=0.26, min=0, max=10, vary=True)
        params.add("gamma", value=1/9.5, vary=False)
        params.add("delta", value=1/11.2, vary=False)
        params.add("alpha", value=0.018, min=0, max=0.2, vary=False)
        params.add("rho", value=1/14, vary=False)
        return params


    def fit(self, data):
        self.train_data = data

        y = data[['total_dead', 'total_infected']].values

        params = self.get_fit_params()

        t = np.arange(len(data))
        minimize_resut = minimize(residual, params, args=(t, data, y, SEIR, self.i0))

        self.fit_result_  = minimize_resut

        best_params = self.fit_result_.params
        for param_name, param_value in best_params.items():
            setattr(self, param_name, param_value)
        return self


    def _predict(self, t, initial_conditions):
        ret = odeint(seir_step, initial_conditions, t, args=(self.params,))
        return ret.T

    def predict_train(self):
        train_data_steps = np.arange(len(self.train_data))
        train_initial_conditions = get_initial_coditions(self, self.train_data)
        return self._predict(train_data_steps, train_initial_conditions)

    def predict_test(self, t):
        S, E, I, R, D = self.predict_train()

        test_initial_conditions = (S[-1], E[-1], I[-1], R[-1], D[-1])
        return self._predict(t, test_initial_conditions)
