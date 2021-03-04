import numpy as np
import lmfit
from lmfit import Parameters, minimize
from copy import deepcopy
from tqdm.auto import tqdm
from .utils import rel_error


def smape_resid_transform(true, pred, eps=1e-5):
    return (true - pred) / (np.abs(true) + np.abs(pred) + eps)


class BaseFitter:
    def __init__(self,
                 result=None,
                 verbose=True,
                 max_iters=None,
                 brute_params=None,
                 save_params_every=100,
                 resid_transform=smape_resid_transform):
        self.result = result
        self.verbose = verbose
        self.max_iters = max_iters
        self.brute_params = brute_params or []
        self.resid_transform = resid_transform
        self.save_params_every = save_params_every
        self.params_history = []
        self.error_history = []

    def residual(self, params, t_vals, data, model):
        pass

    def optimize(self, params, t, data, model, *args, **kwargs):
        with tqdm(total=self.max_iters) as pbar:
            def callback(params, iter, resid, *args, **kwargs):
                if iter % 10 == 0:
                    pbar.n = iter
                    pbar.refresh()
                    pbar.set_postfix({"Error": np.abs(resid).mean()})

                if iter % self.save_params_every == 0 and iter > 0:
                    error = np.abs(resid).mean()
                    self.params_history.append(deepcopy(params))
                    self.error_history.append(error)

            minimize_result = minimize(self.residual,
                                      params,
                                      *args,
                                      args=(t, data, model),
                                      iter_cb=callback,
                                      max_nfev=self.max_iters,
                                      **kwargs)

        self.error_history.append(np.abs(minimize_result.residual).mean())
        self.params_history.append(minimize_result.params)
        return minimize_result

    def optimize_brute(self, params, param_name, brute_params, t, data, model, *args, **kwargs):
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
                result = self.optimize_brute(temp_params, brute_params[0], brute_params[1:], t, data, model, *args, **kwargs)
            else:
                result = self.optimize(temp_params, t, data, model, *args, **kwargs)

            if not best_result or np.mean(np.abs(result.residual)) < np.mean(np.abs(best_result.residual)):
                best_result = result

            # Start optimization from last best point
            last_params = deepcopy(result.params)

        return best_result

    def fit(self, model, data, *args, **kwargs):
        params = model.get_fit_params(data)
        t = np.arange(len(data))

        if not self.brute_params:
            self.result = self.optimize(params, t, data, model, *args, **kwargs)
        else:
            self.result = self.optimize_brute(params, self.brute_params[0], self.brute_params[1:], t, data, model, *args, **kwargs)

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

    def residual(self, params, t_vals, data, model):
        model.params = params

        initial_conditions = model.get_initial_conditions(data)

        (S, E, I, R, D), history = model.predict(t_vals, initial_conditions, history=True)
        new_exposed, new_infected, new_recovered, new_dead = model.compute_daily_values(S, E, I, R, D)
        true_daily_cases = data[self.new_cases_col][:len(new_infected)]
        true_daily_deaths = data[self.new_deaths_col][:len(new_dead)]

        resid_I_new = self.resid_transform(true_daily_cases, new_infected)
        resid_D_new = self.resid_transform(true_daily_deaths, new_dead)

        residuals = np.concatenate([
            resid_I_new,
            resid_D_new,
        ]).flatten()
        return residuals


class HiddenCurveFitter(CurveFitter):
    def residual(self, params, t_vals, data, model):
        model.params = params

        initial_conditions = model.get_initial_conditions(data)

        (S, E, I, Iv, R, Rv, D, Dv), history = model.predict(t_vals, initial_conditions, history=True)
        (new_exposed,
         new_infected_invisible, new_infected_visible,
         new_recovered_invisible,
         new_recovered_visible,
         new_dead_invisible, new_dead_visible) = model.compute_daily_values(S, E, I, Iv, R, Rv, D, Dv)
        true_daily_cases = data[self.new_cases_col][:len(new_infected_visible)].values
        true_daily_deaths = data[self.new_deaths_col][:len(new_dead_visible)].values


        resid_I_new = self.resid_transform(true_daily_cases, new_infected_visible)
        resid_D_new = self.resid_transform(true_daily_deaths, new_dead_visible)

        # print('True', true_daily_cases[:3])
        # print('Forecast', new_infected_invisible[:3])
        # print('Resids', resid_I_new[:3])
        #
        # print('Deaths True', true_daily_deaths[:3])
        # print('Forecast', new_dead_visible[:3])
        # print('Resids', resid_D_new[:3])
        residuals = np.concatenate([
            resid_I_new,
            resid_D_new,
        ]).flatten()
        return residuals
