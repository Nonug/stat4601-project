from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd


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
