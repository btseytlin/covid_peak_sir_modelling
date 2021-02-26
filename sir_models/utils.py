import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm


def compute_daily_values(S, E, I, R, D):
    new_dead = np.diff(D)
    new_recovered = np.diff(R)
    new_infected = np.diff(I) + new_recovered + new_dead
    new_exposed = np.diff(S[::-1])[::-1]

    return new_exposed, new_infected, new_recovered, new_dead


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


def stepwise_soft(t, coefficients):
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
    r = 20
    c = 0.5
    out = sigmoid(t, prev_index, index, q0, q1, c, r)
    return out


def eval_one_day_ahead(df, model_cls, fitter_cls, eval_period_start, n_eval_points=100):
    eval_df = df[df.date >= eval_period_start].iloc[::len(df) // n_eval_points]
    pred_dates = []
    true_D = []
    baseline_pred_D = []
    model_pred_D = []

    for row in tqdm(eval_df.itertuples(), total=len(eval_df)):
        train_df = df.loc[:row.Index]

        pred_dates.append(row.date)
        prev_day = train_df.iloc[-1]
        pred_D = prev_day.total_dead

        model = model_cls()
        fitter = fitter_cls(verbose=False)
        fitter.fit(model, train_df)

        train_initial_conditions = fitter.get_initial_conditions(model, train_df)
        train_t = np.arange(len(train_df))
        (S, E, I, R, D), history = model.predict(train_t, train_initial_conditions)

        test_initial_conds = (S[-1], E[-1], I[-1], R[-1], D[-1])
        (S, E, I, R, D), history = model.predict([train_t[-1], train_t[-1] + 1], test_initial_conds)

        model_pred_D.append(D[-1])
        baseline_pred_D.append(pred_D)
        true_D.append(row.total_dead)

    baseline_D_mae = mean_absolute_error(true_D, baseline_pred_D)

    model_D_mae = mean_absolute_error(true_D, model_pred_D)

    print('Baseline D mae', round(baseline_D_mae, 3))
    print('Model D mae', round(model_D_mae, 3))
    return baseline_D_mae, model_D_mae


