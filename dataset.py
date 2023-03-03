from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor
from collections import defaultdict
from tqdm import tqdm
import os
import random
import torch
import librosa
import numpy as np
import pandas as pd
from util import bs_handler, MOUTH_BS
import pickle



def train_val_test_index(n=1000, train=0.8, validation=0.1, test=0.1):
    random.seed(2023)
    ind = np.arange(n)
    random.shuffle(ind)
    a = int(n*train)
    b = int(n*(train+validation))
    assert(train+validation+test==1 and n>2)
    return set(ind[:a]), set(ind[a:b]), set(ind[b:])

class collater():
    def __init__(self, y_dim, max_len_x, max_len_y):
        self.y_dim = y_dim
        self.max_len_x = max_len_x
        self.max_len_y = max_len_y
    def __call__(self, batch):
        batch_size = 0
        longest_x = 0
        longest_y = 0
        for x,y in batch:
            if x.shape[-1] > longest_x:
                longest_x = x.shape[-1]
                longest_y = y.shape[0]
                
                assert(self.y_dim==y.shape[-1])
            batch_size+=1
        longest_x = min(longest_x, self.max_len_x)
        longest_y = min(longest_y, self.max_len_y)
        out_x_arr = np.zeros((batch_size, 1, longest_x))
        out_y_arr = np.zeros((batch_size, longest_y, self.y_dim))
        for i,(x,y) in enumerate(batch):
            # x, (1, audiolength) y, (csv_length, 31)
            x_leng = min(longest_x,x.shape[-1])
            y_leng = min(longest_y,y.shape[0])
            out_x_arr[i, 0, :x_leng] = x[0, :x_leng]
            out_y_arr[i, :y_leng, :] = y[:y_leng, :]
        return torch.FloatTensor(out_x_arr), torch.FloatTensor(out_y_arr)

"""def collate_fn(batch, y_dim=31):
    batch_size = len(batch)
    longest_x = 0
    longest_y = 0
    out_x_lst = []
    out_y_lst = []
    for x,y in batch:
        if x.shape[-1] > longest_x:
            longest_x = x.shape[-1]
            longest_y = y.shape[0]
            assert(y_dim==y.shape[-1])
    out_x_arr = np.zeros((batch_size, 1, longest_x))
    out_y_arr = np.zeros((batch_size, longest_y, y_dim))
    for i,(x,y) in enumerate(batch):
        # x, (1, audiolength)
        out_x_arr[i, 0, :x.shape[-1]] = x

        out_y_arr[i, :y.shape[0], :] = y
    return torch.FloatTensor(out_x_lst), torch.FloatTensor(out_y_lst)"""

class Wav2BsDataset(Dataset):
    def __init__(self,data):
        self.data = data
        self.n = len(self.data)

    def __getitem__(self, index):
        x = self.data[index]["audio"]
        y = self.data[index]["bs"]
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    def __len__(self):
        return self.n



def read_data(args):
    print("Loading data")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []
    audio_root_path = os.path.join(args.data_path,args.wav_path)
    bs_root_path = os.path.join(args.data_path,args.bs_path)
    
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.base_model_path)
    
    train_ind, val_ind, test_ind = train_val_test_index(n=args.num_wavs)
    dataset_list = args.dataset_list.split(";")
    assert 282 in test_ind, "Random state error"
    for r, ds, fs in os.walk(audio_root_path):
        for f in tqdm(fs):
            if f.endswith(".wav"):
                sub_dir = os.path.basename(r)
                if sub_dir not in dataset_list:
                    break
                key = f.replace(".wav","_") + sub_dir

                wav_path = os.path.join(r,f)
                sig, rate = librosa.load(wav_path, sr=args.sampling_rate)
                audio = processor(sig, return_tensors="pt", sampling_rate=rate).input_values # (1, audiolength)
                bs_path = os.path.join(bs_root_path, sub_dir, f.replace(".wav",".csv"))
                assert os.path.exists(bs_path), "Blendshape file not found:"+bs_path

                data[key]["audio"] = audio
                data[key]["dataset_name"] = sub_dir
                data[key]["bs"] = bs_handler(bs_path,calibration=False)   # bslength, 31

    train_key = []
    val_key = []
    test_key = []
    for k, v in data.items():
        key = int(k.split("_")[0])
        if key in [165,168,172,174,185,193,195,200,329,330,529,621,
                   634,796,797,839,846,904,905,911,912,917,927,929,
                   930,937,939,183,850,932,180,186,638,922]:
            continue
        dataset_name = v["dataset_name"]
        if key in train_ind or dataset_name=="yifeng_150" or key in test_ind:
            train_key.append(key)
            train_data.append(v)
        elif key in val_ind:
            valid_data.append(v)
            val_key.append(key)
        else:
            raise KeyError
    return train_data, valid_data, test_data, train_key, val_key, test_key
                


def create_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, train_key, val_key, test_key = read_data(args)
    max_x = int(16000*20)
    max_y = int(60 * 20)
    collate_fn = collater(y_dim=len(MOUTH_BS),max_len_x=max_x, max_len_y=max_y)
    dataset["Train"] = DataLoader(Wav2BsDataset(train_data), batch_size=args.batch_size, shuffle=True,collate_fn=collate_fn)
    dataset["Valid"] = DataLoader(Wav2BsDataset(valid_data), batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)
    dataset["Test"] = DataLoader(Wav2BsDataset(test_data), batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)
    dataset["Train_key"] = train_key
    dataset["Val_key"] = val_key
    dataset["Test_key"] = test_key
    print("Dataset loaded")
    with open(args.dataset_path, 'wb') as outp:
        pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)
    return args.dataset_path

def get_dataloaders(path):
    with open(path, 'rb') as file:
        dataset = pickle.load(file)
    return dataset