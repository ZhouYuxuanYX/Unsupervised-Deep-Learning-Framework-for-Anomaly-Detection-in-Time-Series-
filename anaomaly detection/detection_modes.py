import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
import sys

def arima(error_data, data, prediction):
    data = pd.DataFrame(data)
    error_data = pd.DataFrame(error_data)
    # define the p, d and q parameters to take any value between 0 and 2
    p = d  = q = range(3)

    # generate all different combinations of p, d and q triples
    # itertools.product works as a nested for loop
    pdq = list(itertools.product(p, d, q))

    # generate all different combinations of seasonal p,q and q triplets
    # seasonal_pdq = [(x[0],x[1],x[2],100) for x in pdq]
    # default no seasonal effect
    # fit the model
    best_aic = np.inf
    best_pdq = None
    best_seasonal_pdq = None
    tmp_model = None
    best_mdl = None
    best_res = None
    AICs = []
    for param in pdq:
            try:
                tmp_mdl = sm.tsa.SARIMAX(error_data, order = param, enforce_stationary=True,
                                  enforce_invetibility=True)
                res = tmp_mdl.fit()
                AICs.append((res.aic))
                if res.aic <= best_aic:
                    best_aic = res.aic
                    best_pdq = param
                    best_mdl = tmp_mdl
                    best_res = res
            except:
                print('Unexpected error:', sys.exc_info()[0])
                AICs.append(0)
                continue
    print('Best SARIMAX{}x{} model - AIC:{}'.format(best_pdq, best_seasonal_pdq,best_aic))

    # visualizing result
    best_res.summary()
    best_res.plot_diagnostics()
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(range(0,len(pdq)), AICs, '*-')
    plt.title('AIC values')
    for i in range(len(pdq)):
        ax.annotate(pdq[i], xy=(i, AICs[i]))

    pred = best_res.get_prediction(start=0 , end=len(error_data)-1)
    # maybe we could adjust the confidence interval dynamically, according to the mse niveau of each file
    # 0.02 means 98% confidence interval
    pred_ci = pred.conf_int(0.01)
    pred_ci.iloc[0] = [0,0]

    # must reverse the error back to original data
    ax = data.plot(label='Observed')
    (pred.predicted_mean+prediction).plot(ax=ax, label='Prediction', alpha=0.5)
    plt.legend(loc='best')


    # draw confidence bound (gray)
    # remark: the first confidence interval is extremely large, so we must substitute it
    pred_ci.iloc[0,:] = pred_ci.iloc[1,:]
    LCL =  pred_ci.iloc[:, 0] + prediction
    UCL = pred_ci.iloc[:, 1] + prediction
    plt.fill_between(pred_ci.index,
                     LCL,
                     UCL, color='g', alpha=.25)
    LCL = pd.DataFrame(LCL)
    LCL.columns = [0]
    UCL = pd.DataFrame(UCL)
    UCL.columns = [0]
    # create a mask array for anomaly points
    mask_anomaly = (LCL>data).astype(int)+(UCL<data).astype(int)
    # Dataframe[i] takes the ith column out as pd.Series
    anomaly = mask_anomaly.loc[mask_anomaly[0]==1]*data
    plt.plot(anomaly,'r*')
    # plt.fill_between(pd.DataFrame(prediction)[mask_anomaly].index, ax.get_ylim(), alpha = 0.9, color='r')

