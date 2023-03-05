#import librosa
#sig, rate = librosa.load("282.wav", sr=16000)
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pywt

# 封装成函数
def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0

def wavelet_noising(new_df):
    data = new_df
    data = data.tolist()  # 将np.ndarray()转化为列表
    w = pywt.Wavelet('sym8')  # 选择sym8小波基
    [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)  # 5层小波分解
    length1 = len(cd1)
    length0 = len(data)

    Cd1 = np.array(cd1)
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0/0.6745)*median_cd1
    lamda = sigma * math.sqrt(2.0*math.log(float(length0), math.e))  # 固定阈值计算
    usecoeffs = []
    usecoeffs.append(ca5)  # 向列表末尾添加对象

    # 软硬阈值折中的方法
    a = 0.5
    for k in range(length1):
        if (abs(cd1[k]) >= lamda):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a*lamda)
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if (abs(cd2[k]) >= lamda):
            cd2[k] = sgn(cd2[k])*(abs(cd2[k])-a*lamda)
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k]) >= lamda):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
        else:
            cd3[k] = 0.0

    length4 = len(cd4)
    for k in range(length4):
        if (abs(cd4[k]) >= lamda):
            cd4[k] = sgn(cd4[k]) * (abs(cd4[k]) - a * lamda)
        else:
            cd4[k] = 0.0

    length5 = len(cd5)
    for k in range(length5):
        if (abs(cd5[k]) >= lamda):
            cd5[k] = sgn(cd5[k]) * (abs(cd5[k]) - a * lamda)
        else:
            cd5[k] = 0.0

    usecoeffs.append(cd5)
    usecoeffs.append(cd4)
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)  # 信号重构
    return recoeffs
from audiomentations import LowPassFilter,Compose,HighPassFilter,Gain
import numpy as np

augment = Compose([
    Gain(
        min_gain_in_db = -1.0,
        max_gain_in_db = 8.0,
        p = 1,
    ),
    LowPassFilter(p=1,min_rolloff=6,max_rolloff=6,min_cutoff_freq=2000,max_cutoff_freq=4500),
    HighPassFilter(p=1,min_rolloff=6,max_rolloff=6,min_cutoff_freq=20,max_cutoff_freq=800)
])
# 主函数
# path = "" #数据路径
import numpy as np
from scipy.io import wavfile
# 提取数据
path = "7.wav"#'test_data/tts_2.wav'

sig,rate = librosa.load(path,sr=16000)
"""try:
    sig = sig[:, 0]
except IndexError:
    pass"""
plt.plot(sig)
plt.show()
print(sig[-100:])


sig=np.array(sig).astype(np.float32)
#augmented_samples = speed_numpy(sig,2)
data1 = augment(samples=sig, sample_rate=rate)

'''data = pd.read_csv(path)
data = data.iloc[:, 0]  # 取第一列数据'''
data1[np.abs(data1)<0.015]=0
plt.plot(data1)
plt.show()
print(data1[-100:])
"""
data_denoising = wavelet_noising(data)  # 调用函数进行小波
plt.plot(data_denoising)  # 显示去噪结果
plt.show()
print(np.mean(data_denoising[:100]))"""
wavfile.write("282aug.wav", rate, data1)