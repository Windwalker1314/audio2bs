import os
os.environ['TRANSFORMERS_CACHE'] = './wav2vec_cache'
import random
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from util import linear_interpolation
import torch
import torch.nn as nn
def plot():
    #from transformers import Wav2Vec2Tokenizer,Wav2Vec2Processor,Wav2Vec2Model
    sig,rate = librosa.load("282.wav", sr=16000)
    sig1,rate1 = sf.read("282.wav")
    time= np.arange(0,len(sig))
    time1 = np.arange(0,len(sig1))
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(time, sig)
    plt.ylabel("Amplitude")
    plt.title("audio 1")
    plt.grid() 

    plt.subplot(2, 1, 2)
    plt.plot(time1, sig1, c="g")
    plt.ylabel("Amplitude")
    plt.title("audio 2")
    plt.grid()
    print(len(sig),len(sig1),rate,rate1)
    plt.show()

def test_hubert():
    import torch
    import torch.nn.functional as F
    import soundfile as sf

    from transformers import (
        Wav2Vec2FeatureExtractor,
        HubertModel,
        Wav2Vec2Processor
    )


    model_path="D:\\Data\\conformer\\hubert-large\chinese-hubert-large"
    wav_path="282.wav"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = HubertModel.from_pretrained(model_path)

    # for pretrain: Wav2Vec2ForPreTraining
    # model = Wav2Vec2ForPreTraining.from_pretrained(model_path)

    model = model.to(device)
    model = model.half()
    model.eval()

    wav, sr = librosa.load(wav_path, sr=16000)
    if len(wav.shape)>1:
        wav = np.mean(wav,axis=1)
    input_values = processor(wav, return_tensors="pt",sampling_rate=sr).input_values
    print(input_values.shape)
    input_values = input_values.half()
    print(input_values.shape)
    input_values = input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        last_hidden_state = outputs.last_hidden_state
        print(last_hidden_state.shape)

def test_inter():
    a = torch.randn(1,500,32)
    #print(linear_interpolation(a,10,50,2600).shape)
    print(a.squeeze(0).shape)

def test_dfinter():
    from scipy import interpolate
    x = np.arange(1, 10)
    y = np.exp(-x/3.0)
    f = interpolate.interp1d(x, y)
    print(f(0))

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
    speed = random.uniform(1.0,1.8)
    sig = x.numpy()
    out_x = librosa.effects.time_stretch(sig, rate=speed)
    out_y = np.apply_along_axis(speed_numpy, 1, y.numpy(), speed=speed)
    return out_x, out_y

def test_transformer():
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4,batch_first=True)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
    src = torch.rand(1, 600, 512)
    out = transformer_encoder(src)
    print(out.shape)

def test_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./model", help="Model path")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    test_transformer()