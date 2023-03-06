from arguments import get_common_args,get_train_args,get_informer_args
from audio2bs import Audio2BS
import librosa
import numpy as np
import time

def infer(args):
    # Load model 
    my_model = Audio2BS(args)
    # Load Auido
    audio, rate = librosa.load("./test_audio.wav.wav", sr=args.sampling_rate)
    # Inference
    t1 = time.time()
    output = my_model.inference(audio, rate)
    # Convert to CSV
    t2 = time.time()

    print(int(round((t2-t1)*1000)))
    output_df = my_model.np_to_csv(output, calibration=False)  #pandas dataframe
    output_df.to_csv("output_2.csv",index=False)

if __name__=="__main__":
    args = get_common_args()
    args = get_train_args(args)
    if "Informer" in args.model_name:
        args = get_informer_args(args)
    infer(args)