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

        self._last_resid = None
        self._last_params = None
        self._terminated = False

    def residual(self, params, t_vals, data, model):
        pass

    def optimize(self, params, t, data, model, *args, **kwargs):
        with tqdm(total=self.max_iters) as pbar:
            def callback(params, iter, resid, *args, **kwargs):
                if iter != self.max_iters:
                    self._last_resid = resid
                    self._last_params = params
                    if (iter % 10 == 0):
                        pbar.n = iter
                        pbar.refresh()
                        pbar.set_postfix({"Error": np.abs(resid).mean()})

                    if (iter % self.save_params_every == 0 and iter > 0):
                        error = np.abs(resid).mean()
                        self.params_history.append(deepcopy(params))
                        self.error_history.append(error)
                else:
                    print('Reached max iters')

            minimize_result = minimize(self.residual,
                                      params,
                                      *args,
                                      args=(t, data, model),
                                      iter_cb=callback,
                                      max_nfev=self.max_iters,
                                      **kwargs)

        if not minimize_result.success:
            minimize_result.params = self._last_params
            minimize_result.residual = self._last_resid

        self._last_resid = None
        self._last_params = None
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
                 new_deaths_col='deaths_per_day',
                 new_cases_col='infected_per_day',
                 new_recoveries_col='recovered_per_day',
                 weights=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.new_deaths_col = new_deaths_col
        self.new_cases_col = new_cases_col
        self.new_recoveries_col = new_recoveries_col
        self.weights = weights

    def residual(self, params, t_vals, data, model):
        model.params = params

        initial_conditions = model.get_initial_conditions(data)

        (S, E, I, R, D), history = model.predict(t_vals, initial_conditions, history=True)
        new_exposed, new_infected, new_recovered, new_dead = model.compute_daily_values(S, E, I, R, D)
        true_daily_cases = data[self.new_cases_col].values[1:]
        true_daily_deaths = data[self.new_deaths_col].values[1:]
        true_daily_recoveries = data[self.new_recoveries_col].values[1:]

        resid_I_new = self.resid_transform(true_daily_cases, new_infected)
        resid_D_new = self.resid_transform(true_daily_deaths, new_dead)
        resid_R_new = self.resid_transform(true_daily_recoveries, new_recovered)

        if self.weights:
            residuals = np.concatenate([
                self.weights['I'] * resid_I_new,
                self.weights['D'] * resid_D_new,
                self.weights['R'] * resid_R_new,
            ]).flatten()
        else:
            residuals = np.concatenate([
                resid_I_new,
                resid_D_new,
                resid_R_new,
            ]).flatten()
        return residuals


class HiddenCurveFitter(CurveFitter):
    def __init__(self, *args,
                 new_recoveries_col='recovered_per_day',
                 weights=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.new_recoveries_col = new_recoveries_col
        self.weights = weights

    def residual(self, params, t_vals, data, model):
        model.params = params

        initial_conditions = model.get_initial_conditions(data)

        (S, E, I, Iv, R, Rv, D, Dv), history = model.predict(t_vals, initial_conditions, history=False)
        (new_exposed,
         new_infected_invisible, new_infected_visible,
         new_recovered_invisible,
         new_recovered_visible,
         new_dead_invisible, new_dead_visible) = model.compute_daily_values(S, E, I, Iv, R, Rv, D, Dv)

        new_infected_visible = new_infected_visible
        new_dead_visible = new_dead_visible
        new_recovered_visible = new_recovered_visible

        true_daily_cases = data[self.new_cases_col].values[1:]
        true_daily_deaths = data[self.new_deaths_col].values[1:]
        true_daily_recoveries = data[self.new_recoveries_col].values[1:]

        resid_I_new = self.resid_transform(true_daily_cases, new_infected_visible)
        resid_D_new = self.resid_transform(true_daily_deaths, new_dead_visible)
        resid_R_new = self.resid_transform(true_daily_recoveries, new_recovered_visible)

        if self.weights:
            residuals = np.concatenate([
                self.weights['I'] * resid_I_new,
                self.weights['D'] * resid_D_new,
                self.weights['R'] * resid_R_new,
            ]).flatten()
        else:
            residuals = np.concatenate([
                resid_I_new,
                resid_D_new,
                resid_R_new,
            ]).flatten()

        return residuals
