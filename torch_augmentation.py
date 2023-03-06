import torch
from torch_audiomentations import Compose, Gain, PitchShift, HighPassFilter,LowPassFilter
import random
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
    if torch.randn(1)<0:
        filter = random.uniform(0.0015,0.01)
        audio_samples[torch.abs(audio_samples)<filter] = 0
    
    #audio_samples = torch.rand(size=(1,1, 32000), dtype=torch.float32, device="cpu")

    # Apply augmentation. This varies the gain and polarity of (some of)
    # the audio snippets in the batch independently.
    perturbed_audio_samples = apply_augmentation(audio_samples, sample_rate=sample_rate)
    return perturbed_audio_samples

def gain(audio, sample_rate):
    apply_aug = Compose(
        transforms=[
            Gain(
                min_gain_in_db=8.0,
                max_gain_in_db=12.0,
                p = 1.0
            )
        ]
    )
    aug = apply_aug(audio,sample_rate)
    return aug