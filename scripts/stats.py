from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMAResults
import pandas as pd
import numpy as np


def adf_test(
    df,
    col="temperature",
    alpha=0.05,
):
    if isinstance(df, pd.core.series.Series):
        result = adfuller(df.values, autolag="AIC")
    else:
        result = adfuller(df[col].values, autolag="AIC")
    print(f"1. ADF Statistic: {result[0]}")
    print("2. Critical Values :")
    for key, val in result[4].items():
        print("\t", key, ": ", val)

    print("3. Num Of Lags : ", result[2])
    print("4. Num Of Observations Used For ADF Regression:", result[3])

    if result[1] < alpha:
        print(f"5. P-value: {result[1]} < {alpha}, H0 rej")
        print("\tTime series is stationary")
    else:
        print(f"5. P-value: {result[1]} > {alpha}, H0 not rej")
        print("\tTime series is not stationary")

    return result


# TODO fix tuple out of range
def kpss_test(
    df,
    col="temperature",
    alpha=0.05,
):
    if isinstance(df, pd.core.series.Series):
        result = kpss(df.values, regression="c", nlags="auto")
    else:
        result = kpss(df.col.values, regression="c", nlags="auto")

    print(f"1. KPSS Statistic: {result[0]}")
    print("2. Critical Values :")
    for key, val in result[3].items():
        print("\t", key, ": ", val)

    print("3. Num Of Lags : ", result[2])

    if result[1] < alpha:
        print(f"p-value: {result[1]} < {alpha}, H0 rej")
        print("\tTime series is not stationary")
    else:
        print(f"p-value: {result[1]} > {alpha}, H0 not rej")
        print("\tTime series is stationary")

    return result


def forecast_accuracy(fitted: ARIMAResults, actual: pd.Series):
    forecast = fitted.get_forecast(len(actual))
    fc = forecast.predicted_mean

    # Make as pandas series
    mape = np.mean(np.abs(fc - actual) / np.abs(actual))  # MAPE
    me = np.mean(fc - actual)  # ME
    mae = np.mean(np.abs(fc - actual))  # MAE
    mpe = np.mean((fc - actual) / actual)  # MPE
    rmse = np.mean((fc - actual) ** 2) ** 0.5  # RMSE
    corr = np.corrcoef(fc, actual)[0, 1]  # corr
    mins = np.amin(np.hstack([fc[:, None], actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([fc[:, None], actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins / maxs)  # minmax
    # acf1 = acf(fc - test)[1]  # ACF1
    return {
        "mape": mape,
        "me": me,
        "mae": mae,
        "mpe": mpe,
        "rmse": rmse,
        # "acf1": acf1,
        "corr": corr,
        "minmax": minmax,
    }
