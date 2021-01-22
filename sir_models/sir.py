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

def get_initial_coditions(population, i0):
    I0 = i0
    Rec0 = 0
    S0 = population - I0 - Rec0
    D0 = 0
    return (S0, I0, Rec0, D0)


def residual(params, t, target, model_class):
    initial_conditions = get_initial_coditions(params['population'],
                                              params['i0'])
    model = model_class(population=params['population'],
                        beta=params['beta'],
                        gamma=params['gamma'],
                        alpha=params['alpha'],
                        rho=params['rho'])
    S, I, R, D = model._predict(t, initial_conditions)

    residuals = np.concatenate([
            D - target[:, 0],
            (I.cumsum() - target[:, 1]),
            # 0.1*(R - target[:, 2]),
        ]).flatten()

    return residuals

# S -> I -> R 
#        -> D

class SIROneStain:
    def __init__(self, population, 
                        beta=None, # Infection rate
                        gamma=None, # Recovery rate
                        alpha=None, # Fatality rate
                        rho=None, # 22 days exposed -> dead
                        i0=None, # Number of infected on day 0
                ):
        self.population = population
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.rho = rho
        self.i0 = i0

        self.fit_result_ = None

        self.train_data = None


    def get_fit_params(self):
        params = Parameters()
        params.add("population", value=self.population, vary=False)
        params.add("beta", value=1.2, min=0, max=10, vary=True)
        params.add("gamma", value=1/11, min=0, max=1, vary=False)
        params.add("alpha", value=0.018, min=0, max=0.2, vary=True)
        params.add("rho", value=1/12, min=0, max=1/12, vary=False)
        params.add("i0", value=1000, min=0, max=self.population, vary=True)

        return params

    def fit(self, data):
        self.train_data = data

        y = data[['total_dead', 'total_infected']].values
        
        params = self.get_fit_params()

        t = np.arange(len(data))
        minimize_resut = minimize(residual, params, args=(t, y, SIROneStain))

        self.fit_result_  = minimize_resut

        best_params = self.fit_result_.params
        self.beta = best_params['beta']
        self.gamma = best_params['gamma']
        self.alpha = best_params['alpha']
        self.rho = best_params['rho']
        self.i0 = best_params['i0']
        return self


    def _predict(self, t, initial_conditions):
        ret = odeint(sir_step_one_stain, initial_conditions, t, 
           args=(self.population, self.beta, self.gamma, self.alpha, self.rho))
        return ret.T
        #return sir_simulate(initial_conditions, t, self.population, self.beta, self.gamma, self.alpha, self.rho)

    def predict_train(self):
        train_data_steps = np.arange(len(self.train_data))

        train_initial_conditions = get_initial_coditions(self.population, self.i0)
        return self._predict(train_data_steps, train_initial_conditions)

    def predict_test(self, t):
        S, I, R, D = self.predict_train()

        test_initial_conditions = (S[-1], I[-1], R[-1], D[-1])
        return self._predict(t, test_initial_conditions)
