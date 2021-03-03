import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm
from matplotlib import pyplot as plt


def shift(arr, num, fill_value=np.nan):
    if num >= 0:
        return np.concatenate((np.full(num, fill_value), arr[:-num]))
    else:
        return np.concatenate((arr[-num:], np.full(-num, fill_value)))


def stepwise(t, coefficients):
    t_arr = np.array(list(coefficients.keys()))

    min_index = np.min(t_arr)
    max_index = np.max(t_arr)

    if t <= min_index:
        index = min_index
    elif t >= max_index:
        index = max_index
    else:
        index = np.min(t_arr[t_arr >= t])
    return coefficients[index]


def sigmoid(x, xmin, xmax, a, b, c, r):
    x_scaled = (x - xmin) / (xmax - xmin)
    out = (a * np.exp(c * r) + b * np.exp(r * x_scaled)) / (np.exp(c * r) + np.exp(x_scaled * r))
    return out


def stepwise_soft(t, coefficients, r=20, c=0.5):
    t_arr = np.array(list(coefficients.keys()))

    min_index = np.min(t_arr)
    max_index = np.max(t_arr)

    if t <= min_index:
        return coefficients[min_index]
    elif t >= max_index:
        return coefficients[max_index]
    else:
        index = np.min(t_arr[t_arr >= t])

    if len(t_arr[t_arr < index]) == 0:
        return coefficients[index]
    prev_index = np.max(t_arr[t_arr < index])
    # sigmoid smoothing
    q0, q1 = coefficients[prev_index], coefficients[index]
    out = sigmoid(t, prev_index, index, q0, q1, c, r)
    return out


def eval_one_day_ahead(df, model_cls, fitter_cls, eval_period_start, n_eval_points=100, total_dead_col='total_deaths',
                       model_kwargs=None, fitter_kwargs=None, deaths_index_in_compartments=4):
    model_kwargs = model_kwargs or {}
    fitter_kwargs = fitter_kwargs or {'verbose': False}

    eval_df = df[df.date >= eval_period_start].iloc[::len(df) // n_eval_points]
    pred_dates = []
    true_D = []
    baseline_pred_D = []
    model_pred_D = []

    progress_bar = tqdm(eval_df.itertuples(), total=len(eval_df))
    for row in progress_bar:
        train_df = df.loc[:row.Index-1]
        pred_dates.append(row.date)
        prev_day = train_df.iloc[-1]
        pred_D = prev_day[total_dead_col]
        model = model_cls(**model_kwargs)
        fitter = fitter_cls(**fitter_kwargs)
        fitter.fit(model, train_df)

        train_initial_conditions = fitter.get_initial_conditions(model, train_df)
        train_t = np.arange(len(train_df))
        state, history = model.predict(train_t, train_initial_conditions)

        test_initial_conds = [compartment[-1] for compartment in state]
        state, history = model.predict([train_t[-1], train_t[-1] + 1], test_initial_conds)

        D = state[deaths_index_in_compartments]
        model_pred_D.append(D[-1])
        baseline_pred_D.append(pred_D)
        true_D.append(getattr(row, total_dead_col))

        model_D_mae = round(mean_absolute_error(true_D, model_pred_D), 3)
        baseline_D_mae = round(mean_absolute_error(true_D, baseline_pred_D), 3)
        progress_bar.set_postfix({"Baseline MAE": baseline_D_mae, "Model MAE": model_D_mae})

    baseline_D_mae = mean_absolute_error(true_D, baseline_pred_D)

    model_D_mae = mean_absolute_error(true_D, model_pred_D)

    print('Baseline D mae', round(baseline_D_mae, 3))
    print('Model D mae', round(model_D_mae, 3))
    return baseline_D_mae, model_D_mae


