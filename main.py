from arguments import get_common_args,get_train_args
from inference import inference, load_model, load_base_model
from transformers import Wav2Vec2FeatureExtractor
from audio2bs import Audio2BS
import librosa
import os

def infer(args):
    
    base_model_path = args.base_model_path
    model_path = os.path.join(args.model_path,args.model_name+".pth")
    device = args.device
    my_model = Audio2BS(base_model_path, model_path, device)
    audio, rate = librosa.load("./test_data/tts.wav", sr=args.sampling_rate)
    output = my_model.inference(audio, rate)
    output_df = my_model.np_to_csv(output, calibration=False)  #pandas dataframe
    output_df.to_csv("output.csv",index=False)

if __name__=="__main__":
    args = get_common_args()
    args = get_train_args(args)
    infer(args)