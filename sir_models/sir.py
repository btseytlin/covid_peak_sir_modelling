import numpy as np
import lmfit
from scipy.integrate import odeint
from lmfit import Parameters, minimize


def sir_step_one_stain(initial_conditions, t, population, beta, gamma, alpha, rho):
    S, I, R, D = initial_conditions

    new_infected = beta * I * (S / population)
    new_dead = alpha * rho * I
    new_recovered = gamma * (1-alpha) * I 

    
    dSdt = -new_infected
    dIdt = new_infected - new_recovered - new_dead
    dRdt = new_recovered
    dDdt = new_dead

    assert S + I + R + D - population <= 1e10, (S + I + R + D - population)
    assert dSdt + dIdt + dRdt + dDdt <= 1e10, (dSdt, dIdt, dRdt, sum((dSdt, dIdt, dRdt)))
    return dSdt, dIdt, dRdt, dDdt


def sir_step(initial_conditions, population, beta, gamma, alpha, rho):
    assert (np.array(initial_conditions) >= 0).all(), np.array(initial_conditions)

    S, I, R, D = initial_conditions

    new_infected = beta * I * (S / population)
    new_dead = alpha * rho * I
    new_recovered = gamma * (1-alpha) * I 

    
    dSdt = -new_infected
    dIdt = new_infected - new_recovered - new_dead
    dRdt = new_recovered
    dDdt = new_dead

    new_S = S + dSdt
    new_I = I + dIdt
    new_R = R + dRdt
    new_D = D + dDdt
    #print(S, dSdt, new_S)
    assert S + I + R + D - population <= 1e10, (S + I + R + D - population)
    assert new_S + new_I + new_R + new_D - population <= 1e10, ( new_S + new_I + new_R + new_D - population)
    assert dSdt + dIdt + dRdt + dDdt <= 1e10, (dSdt, dIdt, dRdt, sum((dSdt, dIdt, dRdt)))
    return new_S, new_I, new_R, new_D


def sir_simulate(initial_conditions, t, population, beta, gamma, alpha, rho):
    S, I, R, D = initial_conditions

    S_values = [S]
    I_values = [I]
    R_values = [R]
    D_values = [D]

    for step in t[1:]:
        #print('Step', step)
        step_initial = (S_values[-1], I_values[-1], R_values[-1], D_values[-1])
        #print('Initial', step_initial)
        new_S, new_I, new_R, new_D  = sir_step(step_initial, population, beta, gamma, alpha, rho)
        #print('New', new_S, new_I, new_R, new_D)
        S_values.append(new_S)
        I_values.append(new_I)
        R_values.append(new_R)
        D_values.append(new_D)

    S_values = np.array(S_values)
    I_values = np.array(I_values)
    R_values = np.array(R_values)
    D_values = np.array(D_values)

    return S_values, I_values, R_values, D_values


def residual(params, t, target, initial_conditions, model_class):
    model = model_class(population=params['population'],
                        beta=params['beta'],
                        gamma=params['gamma'],
                        alpha=params['alpha'],
                        rho=params['rho'])
    S, I, R, D = model._predict(t, initial_conditions)

    residuals = np.concatenate([
            D - target,
        ]).flatten()

    return residuals


class SIROneStain:
    def __init__(self, population, 
                        beta=None, # Infection rate
                        gamma=None, # Recovery rate
                        alpha=None, # Fatality rate
                        rho=None, # 22 days exposed -> dead
                ):
        self.population = population
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.rho = rho

        self.fit_result_ = None

    def get_initial_coditions(self, infected, total_recovered):
        I0 = infected
        Rec0 = total_recovered
        S0 = self.population - I0 - Rec0
        D0 = 0
        return (S0, I0, Rec0, D0)


    def fit(self, data, y):
        params = Parameters()
        params.add("population", value=self.population, vary=False)
        params.add("beta", value=1.2, min=0, max=3, vary=True)
        params.add("gamma", value=1/23, min=0, max=1, vary=False)
        params.add("alpha", value=0.018, min=0, max=0.1, vary=False)
        params.add("rho", value=1/22, min=0, max=1/12, vary=False)

        initial_conditions = self.get_initial_coditions(data.iloc[0].infected, 
            data.iloc[0].total_recovered)
        t = np.arange(len(data))

        minimize_resut = minimize(residual, params, 
            args=(t, y, initial_conditions, SIROneStain))


        self.fit_result_  = minimize_resut

        best_params = self.fit_result_ .params
        self.beta = best_params['beta']
        self.gamma = best_params['gamma']
        self.alpha = best_params['alpha']
        self.rho = best_params['rho']
        return self


    def _predict(self, t, initial_conditions):
        ret = odeint(sir_step_one_stain, initial_conditions, t, 
           args=(self.population, self.beta, self.gamma, self.alpha, self.rho))
        return ret.T
        #return sir_simulate(initial_conditions, t, self.population, self.beta, self.gamma, self.alpha, self.rho)

    def predict(self, data, steps=None):
        initial_conditions = self.get_initial_coditions(data.iloc[0].infected, data.iloc[0].total_recovered)
        if steps is None:
            steps = len(data)

        t = np.arange(steps)
        return self._predict(t, initial_conditions)
