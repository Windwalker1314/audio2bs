import argparse
import torch

def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="D:\\Data\\conformer\\hubert-large\chinese-hubert-large", help="The hubert model path")
    parser.add_argument("--model_path", type=str, default="./model", help="Model path")
    parser.add_argument("--model_name", type=str, default="LSTM_v2.0", help="Model Name")
    parser.add_argument("--sampling_rate", type=int, default=16000, help="Sampling rate of wav file (resample)")
    parser.add_argument("--audio_section_length", type=float, default=1, help="max_audio_length")
    args = parser.parse_args()
    return args

def get_train_args(args):
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args

def get_server_args(args):
    args.IP = "127.0.0.1"
    args.port = "2890"
    return args