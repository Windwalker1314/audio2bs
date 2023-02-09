from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, AirAbsorption,ClippingDistortion, AddBackgroundNoise, PolarityInversion,Gain
import numpy as np
import pandas as pd
from scipy.io import wavfile
import random
from tqdm.contrib import tzip
from util import MOUTH_BS
from main import np_to_csv
random.seed(20232023)

def speed_numpy(samples, speed):
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    old_length = samples.shape[0]
    new_length = int(old_length / speed)
    old_indices = np.arange(old_length)  # (0,1,2,...old_length-1)
    new_indices = np.linspace(start=0, stop=old_length, num=new_length)  # 在指定的间隔内返回均匀间隔的数字
    samples = np.interp(new_indices, old_indices, samples)  # 一维线性插值
    samples = samples.astype(data_type)
    return samples

def speed_df(csv_path,out_path,speed):
    samples_2d = pd.read_csv(csv_path)[MOUTH_BS].to_numpy()
    prev = samples_2d.shape[0]
    samples_2d = np.apply_along_axis(speed_numpy, 0, samples_2d, speed=speed)
    out_df = np_to_csv(samples_2d,calibration=False).to_csv(out_path,index=False)
    return out_df


def augmentation(wavpath,speed=1,out_path=None,return_wav=False,noise_dir="D:\\Data\\audio2bs\\1000sentences\\noise"):
    #noise = os.listdir(noise_dir)
    #n = len(noise)
    #selected_ind = random.randint(0,n-1)
    #noise_path = os.path.join(noise_dir, noise[selected_ind])
    augment = Compose([
        AddGaussianNoise(min_amplitude=1, max_amplitude=200, p=0.5),
        TimeStretch(min_rate=speed, max_rate=speed, p=1.0,leave_length_unchanged=False),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.8),
        AirAbsorption(p=0.5),
        Gain(min_gain_in_db=0, max_gain_in_db=12.0,p=0.8)
        #ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=20, p=0.5),
        #AddBackgroundNoise(sounds_path=noise_path, min_snr_in_db=3,max_snr_in_db=30,noise_transform=PolarityInversion(),p=1.0)
    ])

    rate, sig = wavfile.read(wavpath)
    try:
        sig = sig[:, 0]
    except IndexError:
        pass
    sig=np.array(sig).astype(np.float32)
    #augmented_samples = speed_numpy(sig,2)
    augmented_samples = augment(samples=sig, sample_rate=rate).astype(np.int16)

    if out_path is not None:
        wavfile.write(out_path, rate, augmented_samples)
    if return_wav:
        return augmented_samples, rate