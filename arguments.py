import argparse
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",choices=["train","eval","test"], help="The probgram mode (train/eval/test)")
    parser.add_argument("--base_model_path", type=str, default="D:\\Data\\conformer\\hubert-large\\chinese-hubert-large", help="The hubert model path")
    parser.add_argument("--model_path", type=str, default="./model/model_checkpoints", help="Model path")
    parser.add_argument("--model_name", type=str, default="LSTM_v2.2", help="Model Name")
    parser.add_argument("--dataset_path", type=str, default="./data/dataset/dataset.pkl", help="Model path")
    parser.add_argument("--dataset_list", type=str, default="yifeng;yifeng_150;yifeng_350;clean_yifeng;clean_yifeng_350", help="dataset name")
    parser.add_argument("--data_path", type=str, default="D:\\Data\\audio2bs\\data", help="The dataset root path")
    parser.add_argument("--wav_path", type=str, default="audio", help="The subdir that stores audio data")
    parser.add_argument("--bs_path", type=str, default="bs", help="The subdir that stors blendshape data")
    parser.add_argument("--test_data_path", type=str, default="./data/test_data", help="The dataset root path")

    parser.add_argument("--create_dataset", type=str2bool, default=False, help="Whether to (re)create the dataset object from data")
    parser.add_argument("--load_model",  type=str2bool, default=True, help="Whether to load the pretrained model and resume training")
    parser.add_argument("--load_optimizer",  type=str2bool, default=True, help="Whether to load the optimizer")
    parser.add_argument("--augmentation", type=str2bool, default=True, help="Whether to use audio augmentation")

    parser.add_argument("--num_wavs", type=int, default=1000, help="The total number of audio data instances")
    parser.add_argument("--epochs", type=int, default=50, help="The number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=1, help="The number of epochs for training")
    parser.add_argument("--patience", type=int, default=10, help="Early Stopping")
    parser.add_argument("--sampling_rate", type=int, default=16000, help="Sampling rate of wav file (resample)")
    parser.add_argument("--output_fps", type=int, default=60, help="Sampling rate of wav file (resample)")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam"], help="learning rate")
    parser.add_argument("--criterion", type=str, default="WMSE", choices=["MSE", "WMSE"], help="criterion, can be MSE or WMSE (weighted MSE))")
    parser.add_argument("--schedular", type=str, default="ReduceLROnPlateau", choices=["ReduceLROnPlateau"], help="learning rate")
    parser.add_argument("--max_audio_length", type=float, default=20, help="max_audio_length")
    parser.add_argument("--audio_section_length", type=float, default=1, help="audio_section_length(for training)")
    
    parser.add_argument("--smoothing_alpha", type=float, default=0,  help="Smoothing parameter")
    args = parser.parse_args()
    return args

def get_train_args(args):
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args

def get_informer_args(args):
    args.enc_in = 1024
    args.dec_in = 31
    args.c_out = 31
    args.max_seq_length=512
    args.factor=5
    args.d_model=256
    args.n_heads=8
    args.n_encoder_layers = 3
    args.n_decoder_layers = 3
    args.d_feedforward = 512
    args.dropout_rate = 0.05
    args.attention_type = "prob"
    args.activation = "gelu"
    return args

def get_LSTM_args(args):
    args.input_size=1024
    args.hidden_layer_size=128
    args.output_size=31
    args.dropout=0.5
    args.n_layers=2
    args.bidirectional=True
    return args
