from arguments import get_common_args,get_train_args,get_informer_args
from audio2bs import Audio2BS
import librosa

def infer(args):
    # Load model 
    my_model = Audio2BS(args)
    # Load Auido
    audio, rate = librosa.load("./test_audio.wav", sr=args.sampling_rate)
    # Inference
    output = my_model.inference(audio, rate)
    # Convert to CSV
    output_df = my_model.np_to_csv(output, calibration=False)  #pandas dataframe
    output_df.to_csv("output_2.csv",index=False)

if __name__=="__main__":
    args = get_common_args()
    args = get_train_args(args)
    if "Informer" in args.model_path:
        args = get_informer_args(args)
    infer(args)