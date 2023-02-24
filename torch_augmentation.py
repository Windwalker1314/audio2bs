import torch
from torch_audiomentations import Compose, Gain, PitchShift, AddBackgroundNoise

def augmentation(audio_samples,sample_rate):
    apply_augmentation = Compose(
        transforms=[
            Gain(
                min_gain_in_db = -1.0,
                max_gain_in_db = 8.0,
                p = 0.9,
            ),
            PitchShift(sample_rate=sample_rate,p=0.9)
        ]
    )
    
    #audio_samples = torch.rand(size=(1,1, 32000), dtype=torch.float32, device="cpu")

    # Apply augmentation. This varies the gain and polarity of (some of)
    # the audio snippets in the batch independently.
    perturbed_audio_samples = apply_augmentation(audio_samples, sample_rate=sample_rate)
    return perturbed_audio_samples