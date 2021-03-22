import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed


def smape(true, pred, eps=1e-5):
    return np.abs((true - pred) / (np.abs(true) + np.abs(pred) + eps)).mean()


def rel_error(true, pred, eps=1e-5):
    return (true - pred) / (np.maximum(true, pred) + eps)


def shift(arr, num, fill_value=np.nan):
    if num >= 0:
        return np.concatenate((np.full(num, fill_value), arr[:-num]))
    else:
        return np.concatenate((arr[-num:], np.full(-num, fill_value)))


def pad_left(arr, n_zeros=1):
    return np.pad(arr, (n_zeros, 0))


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


def eval_on_select_dates_and_k_days_ahead(df,
                                     eval_func,
                                     eval_dates,
                                     k=7,
                                     verbose=True,
                                     n_jobs=-1):


    def process_date(date):
        t = len(df[df.date < date])
        train_df = df.iloc[:t]

        train_t = np.arange(len(train_df))
        eval_t = np.arange(train_t[-1] + 1, t + k, 1)

        model, fitter, test_states = eval_func(train_df, t, train_t, eval_t)

        test_df = df.iloc[eval_t]

        return (train_df, test_df, test_states, fitter, model)

    results = Parallel(n_jobs=n_jobs, verbose=11 if verbose else 0)(delayed(process_date)(date) for date in eval_dates)

    train_dfs = [x[0] for x in results]
    test_dfs = [x[1] for x in results]
    model_predictions = [x[2] for x in results]
    fitters = [x[3] for x in results]
    models = [x[4] for x in results]

    return models, fitters, model_predictions, train_dfs, test_dfs
