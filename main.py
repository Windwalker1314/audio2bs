from arguments import get_common_args,get_train_args
from inference import inference
from model import LSTM
import os

def infer(args):
    model = LSTM()
    model.to(args.device)
    test_data = os.listdir(args.test_data_path)
    wavs = []
    for i in test_data:
        if i.endswith(".wav"):
            wavs.append(i)
    inference(args, model, checkpoint_path=os.path.join(args.model_path, args.model_name+".pth"), wav_lst=wavs, calibration=False)

if __name__=="__main__":
    args = get_common_args()
    args = get_train_args(args)
    infer(args)