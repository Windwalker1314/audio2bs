import numpy as np
import random
import librosa
import torch
from torch_audiomentations import Compose, Gain, PitchShift, HighPassFilter,LowPassFilter

def augmentation(audio_samples,sample_rate):
    apply_augmentation = Compose(
        transforms=[
            Gain(
                min_gain_in_db = -12.0,
                max_gain_in_db = 12.0,
                p = 0.9,
            ),
            PitchShift(sample_rate=sample_rate,p=0.9),
            LowPassFilter(p=0.5,min_cutoff_freq=3000,max_cutoff_freq=4500),
            HighPassFilter(p=0.5,min_cutoff_freq=100,max_cutoff_freq=500)
        ]
    )
    if torch.randn(1)<-0.5:
        filter = random.uniform(0.001,0.002)
        audio_samples[torch.abs(audio_samples)<filter] = 0
    
    #audio_samples = torch.rand(size=(1,1, 32000), dtype=torch.float32, device="cpu")

    # Apply augmentation. This varies the gain and polarity of (some of)
    # the audio snippets in the batch independently.
    perturbed_audio_samples = apply_augmentation(audio_samples, sample_rate=sample_rate)
    return perturbed_audio_samples

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

def speed_changing(x,y):
    speed = random.uniform(1.0,1.5)
    sig = x.numpy()
    out_x = librosa.effects.time_stretch(sig, rate=speed)
    out_y = np.apply_along_axis(speed_numpy, 1, y.numpy(), speed=speed)
    return torch.from_numpy(out_x), torch.from_numpy(out_y)