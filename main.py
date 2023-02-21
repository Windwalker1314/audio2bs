from arguments import get_common_args,get_train_args
from audio2bs import Audio2BS
import librosa
import os

def infer(args):
    base_model_path = args.base_model_path
    model_path = os.path.join(args.model_path,args.model_name+".pth")
    device = args.device
    # Load model 
    my_model = Audio2BS(base_model_path, model_path, device, audio_section_length=args.audio_section_length)
    # Load Auido
    audio, rate = librosa.load("./20230220.wav", sr=args.sampling_rate)
    print(max(audio))
    # Inference
    output = my_model.inference(audio, rate)
    # Convert to CSV
    output_df = my_model.np_to_csv(output, calibration=False)  #pandas dataframe
    output_df.to_csv("output.csv",index=False)

if __name__=="__main__":
    args = get_common_args()
    args = get_train_args(args)
    infer(args)