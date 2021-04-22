# Study: What will be the effect of SARS-COV-2 B.1.1.7? 

![Covid B.1.1.7 scenarios](
https://raw.githubusercontent.com/btseytlin/covid_peak_sir_modelling/main/figures/presentation/scenarios_beta2mult.png)

## Reproducing results:
0. Optionally replace `data/data.csv` with your own dataset of the same format. The original data for Russia was obtained [here](https://yandex.ru/covid19/stat).
1. Run the data preparation [notebook](https://github.com/btseytlin/covid_peak_sir_modelling/blob/main/prepare_data.ipynb)
2. Run [seir_moscow_hidden.ipynb](https://github.com/btseytlin/covid_peak_sir_modelling/blob/main/seir_moscow_hidden.ipynb) to train and evaluate the usual SARS-COV-2 model.
3. Run [seir_two_strain_hidden_states_moscow.ipynb](https://github.com/btseytlin/covid_peak_sir_modelling/blob/main/seir_two_strain_hidden_states_moscow.ipynb) to use that model to forecast for two strains.


Articles:
* [habr.com: Чем грозит Москве «британский» штамм COVID-19? Отвечаем с помощью Python и дифуров](https://m.habr.com/ru/company/otus/blog/553638/)
