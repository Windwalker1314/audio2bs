import argparse
import torch

def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="D:\\Data\\conformer\\hubert-large\chinese-hubert-large", help="The hubert model path")
    parser.add_argument("--model_path", type=str, default="./model", help="Model path")
    parser.add_argument("--dataset_path", type=str, default="./dataset.pkl", help="Model path")
    parser.add_argument("--dataset_list", type=str, default="yifeng;yifeng_150", help="Model path")
    parser.add_argument("--data_path", type=str, default="D:\\Data\\audio2bs\\data", help="The dataset root path")
    parser.add_argument("--test_data_path", type=str, default="./test_data", help="The dataset root path")
    parser.add_argument("--wav_path", type=str, default="audio", help="The subdir that stores audio data")
    parser.add_argument("--bs_path", type=str, default="bs", help="The subdir that stors blendshape data")
    parser.add_argument("--model_name", type=str, default="Conformer", help="Model Name")
    parser.add_argument("--cuda", type=bool, default=True, help="whether to use GPU")
    parser.add_argument("--load_model", type=bool, default=False, help="Whether to load the pretrained model and resume training")
    parser.add_argument("--augmentation", type=bool, default=True, help="Whether to use audio augmentation")
    parser.add_argument("--num_wavs", type=int, default=1000, help="The total number of audio data instances")
    parser.add_argument("--epochs", type=int, default=50, help="The number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=1, help="The number of epochs for training")
    parser.add_argument("--patience", type=int, default=2, help="Early Stopping")
    parser.add_argument("--sampling_rate", type=int, default=16000, help="Sampling rate of wav file (resample)")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--max_audio_length", type=float, default=2, help="max_audio_length")
    parser.add_argument("--audio_section_length", type=float, default=1, help="max_audio_length")
    args = parser.parse_args()
    return args

def get_train_args(args):
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args

