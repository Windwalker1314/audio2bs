# 单指数平滑
import matplotlib.pyplot as plt
import pandas as pd
from util import MOUTH_BS

data_LSTM = pd.read_csv("./test_data/cctv1_3_LSTM_aug.csv")
data_conformer = pd.read_csv("./test_data/cctv1_3_Conformer.csv")

def exponential_smoothing(series, alpha):
    """
        series - dataset with timestamps
        alpha - float [0.0, 1.0], smoothing parameter
    """
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result
    
def plotExponentialSmoothing(series1, series2, alphas):
    """
        Plots exponential smoothing with different alphas
        
        series - dataset with timestamps
        alphas - list of floats, smoothing parameters
        
    """
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(25, 7))
        for alpha in alphas:
            plt.plot(exponential_smoothing(series1, alpha), label="Alpha {}".format(alpha))
        plt.plot(series1.values, "c", label = "Conformer")
        plt.plot(series2.values, "r", label = "LSTM")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True)
        plt.show()


def double_exponential_smoothing(series, alpha, beta):
    print(series)
    """
        series - dataset with timeseries
        alpha - float [0.0, 1.0], smoothing parameter for level
        beta - float [0.0, 1.0], smoothing parameter for trend
    """
    # first value is same as series
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result
 
def plotDoubleExponentialSmoothing(series1, series2, alphas, betas):
    """
        Plots double exponential smoothing with different alphas and betas
        
        series - dataset with timestamps
        alphas - list of floats, smoothing parameters for level
        betas - list of floats, smoothing parameters for trend
    """
    
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(13, 5))
        for alpha in alphas:
            for beta in betas:
                plt.plot(double_exponential_smoothing(series1, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
        plt.plot(series1.values, label = "Conv+transformer")
        plt.plot(series2.values, label = "LSTM")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True)
        plt.show()

#plotExponentialSmoothing(data_conformer['JawOpen'], data_LSTM['JawOpen'], alphas=[0.3],)   
#plotDoubleExponentialSmoothing( data_conformer['JawOpen'][:200], data_LSTM['JawOpen'][:200], alphas=[0.5,0.3], betas=[0.9,0.3])

df_out = data_conformer[MOUTH_BS].apply(exponential_smoothing, alpha=0.3,axis=0)
df_out.insert(0, "BlendShapeCount", data_conformer["BlendShapeCount"])
df_out.insert(0, "Timecode",  data_conformer["Timecode"])
#print(df_out.head())
df_out.to_csv("post_output.csv",index=False)