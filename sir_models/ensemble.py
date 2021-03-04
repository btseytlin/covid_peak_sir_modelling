import numpy as np
from scipy.special import softmax
from tqdm.auto import tqdm


class EnsembleModel:
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.array(weights)

        self.fitter = None

    @classmethod
    def train(cls, model, fitter, data):
        fitter.fit(model, data)

        weights = softmax(1 - np.array(fitter.error_history))
        ensemble_model = cls([model.__class__(params=params) for params in fitter.params_history], weights)
        ensemble_model.fitter = fitter

        return ensemble_model

    def predict(self, data=None, initial_conditions=None, t=None, history=True):
        t = t if t is not None else np.arange(len(data))
        model_daily_vals = []
        weighted_model_daily_vals = []
        model_states = []
        weighted_model_states = []
        for i, model in tqdm(enumerate(self.models)):
            model_initial_conditions = initial_conditions if initial_conditions is not None else model.get_initial_conditions(data)

            state, model_history = model.predict(t, model_initial_conditions, history=history)
            daily_vals = model.compute_daily_values(*state)
            model_states.append(np.array(state))
            weighted_model_states.append(self.weights[i] * np.array(state))

            model_daily_vals.append(daily_vals)
            weighted_model_daily_vals.append(self.weights[i] * np.array(daily_vals))

        aggregate_states = np.sum(weighted_model_states, axis=0)
        aggregate_daily = np.sum(weighted_model_daily_vals, axis=0)

        return aggregate_states, aggregate_daily, np.array(model_states), np.array(model_daily_vals), model_history

