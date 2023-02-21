from arguments import get_common_args,get_train_args
from audio2bs import Audio2BS
import librosa
import os
import numpy as np
import time
def load_pcm(pcm_file):
    b = np.fromfile(pcm_file,dtype=np.int16)
    return b

def infer(args):
    base_model_path = args.base_model_path
    model_path = os.path.join(args.model_path,args.model_name+".pth")
    device = args.device
    # Load model 
    my_model = Audio2BS(base_model_path, model_path, device, audio_section_length=args.audio_section_length)
    # Load Auido
    audio, rate = librosa.load("./test_data/tts_1.wav", sr=args.sampling_rate)
    # Inference
    t1 = time.time()
    output = my_model.inference(audio, rate)
    # Convert to CSV
    t2 = time.time()

    print(int(round((t2-t1)*1000)))
    output_df = my_model.np_to_csv(output, calibration=False)  #pandas dataframe
    output_df.to_csv("output.csv",index=False)

if __name__=="__main__":
    args = get_common_args()
    args = get_train_args(args)
    infer(args)