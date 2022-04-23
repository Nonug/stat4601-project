import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.tsa.arima.model import ARIMAResults



def plot_df(
    df, x="", y="", title="", xlabel="Date", ylabel="Value", dpi=100, figsize=(30, 10)
):
    plt.figure(figsize=figsize, dpi=dpi)
    # plt.plot(x, y, color='tab:red')
    if isinstance(df, pd.core.series.Series):
        sns.lineplot(data=df)
    else:
        sns.lineplot(data=df, x=x, y=y)
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)


def plot_acf_pacf(series):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=100)
    plot_acf(series.tolist(), lags=50, ax=axes[0])
    plot_pacf(series.tolist(), lags=50, ax=axes[1])
    plt.show()


def plot_seasonal_decompose(
    result: DecomposeResult,
    dates: pd.Series = None,
    title: str = "Seasonal Decomposition",
):
    x_values = dates if dates is not None else np.arange(len(result.observed))
    return (
        make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.observed, mode="lines", name="Observed"),
            row=1,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.trend, mode="lines", name="Trend"),
            row=2,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.seasonal, mode="lines", name="Seasonal"),
            row=3,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.resid, mode="lines", name="Residual"),
            row=4,
            col=1,
        )
        .update_layout(
            height=900,
            title=f"<b>{title}</b>",
            margin={"t": 100},
            title_x=0.5,
            showlegend=False,
        )
    )

def plot_forecast(fitted: ARIMAResults, test : pd.Series, _title = "Forecast vs Actual"):
    forecast = fitted.get_forecast(len(test))
    fc = forecast.predicted_mean
    conf = forecast.conf_int(alpha=0.05) # 95% conf

    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf.iloc[:, 0], index=test.index)
    upper_series = pd.Series(conf.iloc[:, 1], index=test.index)

    # Plot
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(fitted.data.orig_endog[:], label='training')
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series, 
                    color='k', alpha=.15)
    plt.title(_title)
    plt.legend(loc='upper left', fontsize=8)
    plt.show()