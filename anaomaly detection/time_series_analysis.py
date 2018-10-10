import pandas as pd
from statsmodels.tsa.stattools import adfuller
from ts_plots import  scatter_plot_of_lags, plot_rolling_average, ts_plot, decomposition_plot

def adf_test(data):
    """
    Perform Augumented Dickey Fuller test(unit root test):
    test null-hypothesis for 'non-stationary random walk process'
    """
    dftest = adfuller(data, regression='ctt',autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value ({})'.format(key)] = value
    print(dfoutput)

def ts_analysis(data):
    # if time series is not a Series object, convert it to
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    # apply time series analysis to the data
    adf_test(data)
    ts_plot(data, 10)
    scatter_plot_of_lags(data, 9)

    # apply time series analysis to the 1 order differenced data
    data_diff1 = data.diff().fillna(0)
    adf_test(data_diff1)
    ts_plot(data_diff1,10)
    scatter_plot_of_lags(data_diff1, 9)

    # apply rolling average to observe the trend
    plot_rolling_average(data, 12)

    # Decomposition of the original signal
    decomposition_plot(data, period=10)