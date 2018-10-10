import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import statsmodels.graphics.gofplots as smg
import math
from statsmodels.tsa.seasonal import seasonal_decompose
from stldecompose import decompose
from scipy.stats.distributions import t

def ts_plot(y, lags=None, title=''):
    """Calculate acf, pacf, histogram, and qq plot for a given time sereis
    """
    # if time series is not a Series object, convert it to
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # initialize figure and axes
    fig = plt.figure(figsize=(14, 12))
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan = 2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    hist_ax = plt.subplot2grid(layout, (2, 1))

    # time series plot
    y.plot(ax=ts_ax)
    plt.legend(loc="best")
    ts_ax.set_title(title)

    # acf and pacf plot
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)

    # qq plot
    smg.qqplot(y, line='s', dist=t, fit=True,ax=qq_ax)
    qq_ax.set_title('Normal QQ Plot')

    # hist plot
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    plt.tight_layout()
    # Remark: ion() turns the interactive mode on, unless the program will be blocked when the figure shows
    # But is only used for debug and interactive mode
    plt.ion()
    plt.show()
    plt.pause(0.01)
    return

def scatter_plot_of_lags(series_data, lags):
    """
    automatically adjust the layout of subplots to the input lags

    Args:
        series_data: pandas Series object
        lags: number lags to be plotted
    """
    # if time series is not a Series object, convert it to
    if not isinstance(series_data, pd.Series):
        series_data = pd.Series(series_data)

    ncols = 3
    # calculate the layout of subplots
    nrows = math.ceil(lags/ncols)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols, 5 * nrows))

    for ax, lag in zip(axes.flat, np.arange(1, lags + 1, 1)):
        # create the title string automatically according to the lag
        lag_str = 't-{}'.format(lag)
        # concatenate the lagged series to a Pandas DataFrame object
        X = (pd.concat([series_data, series_data.shift(-lag)], axis=1, keys=['y'] + [lag_str]))

        # plot data
        X.plot(ax=ax, kind='scatter', y='y', x=lag_str)
        # use the DataFrame method to get the correlation
        corr = X.corr().values[0][1]
        ax.set_ylabel('Original')
        ax.set_title('Lag: {} (corr={:.2f}'.format(lag_str, corr))
        ax.set_aspect('equal')

    fig.tight_layout()
    # Remark: ion() turns the interactive mode on, unless the program will be blocked when the figure shows
    # But is only used for debug and interactive mode
    plt.ion()
    plt.show()
    plt.pause(0.01)

def plot_rolling_average(series_data, window=12):
    """
    Plot rolling mean and rolling standard deviation for a given time series and window

    Args:
        series_data: pandas Series object
        lags: number lags to be plotted
    """
    # if time series is not a Series object, convert it to
    if not isinstance(series_data, pd.Series):
        series_data = pd.Series(series_data)
    # calculate moving averages
    rolling_mean = series_data.rolling(window).mean()
    # median is more robust to outliers
    # rolling_median = data.rolling(window).median()
    rolling_std = series_data.rolling(window).std()

    # plot statistics
    plt.figure()
    plt.plot(series_data, label='Original')
    plt.plot(rolling_mean, color='crimson', label='Moving average mean')
    plt.plot(rolling_std, color='darkslateblue', label='Moving average standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean % Standard Deviation')
    plt.ion()
    plt.show()
    plt.pause(0.01)
    return

def decomposition_plot(series_data, period):
    """
    decomposition of the original signal for preliminary analysis

    Args:
        series_data: Pandas Series object
        period: estimated seasonal frequency
    """
    # if time series is not a Series object, convert it to
    if not isinstance(series_data, pd.Series):
        series_data = pd.Series(series_data)
    # naive additive decomposition
    decomp = seasonal_decompose(series_data.values, model='additive', freq=period)
    decomp.plot()

    # stl decompose
    stl = decompose(series_data.values, period=period)
    stl.plot()
    plt.show()
    plt.pause(0.01)




