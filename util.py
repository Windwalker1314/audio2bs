import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch

MOUTH_BS = ['JawForward', 'JawRight', 'JawLeft', 'JawOpen',
       'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthRight', 'MouthLeft',
       'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft',
       'MouthFrownRight', 'MouthDimpleLeft', 'MouthDimpleRight',
       'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower',
       'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper',
       'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft',
       'MouthLowerDownRight', 'MouthUpperUpLeft', 'MouthUpperUpRight',
       'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight','TongueOut']

MOUTH_BS_WEIGHT = [0.5, 0.2, 0.2, 1.5,
       1.5, 1.5, 1.5, 1, 1,
       1, 1, 1,
       1, 1, 1,
       1, 1, 1,
       1, 1, 1,
       1, 1, 1,
       1, 1, 1,
       0.5, 0.2, 0.2, 1]

class EarlyStopping():
    def __init__(self, model_name, best_score=-np.Inf, patience=4, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_name= model_name
    def __call__(self,val_loss,model,optimizer,path):
        if self.verbose:
            print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,optimizer,path)
        elif score < self.best_score+self.delta:
            self.counter+=1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model,optimizer,path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, optimizer, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, path)
        #torch.save(model.state_dict(), path+'/'+model_name+'.pth')
        self.val_loss_min = val_loss

class Weighted_MSE(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer('w', torch.FloatTensor(weight))
        
    def forward(self, y_pred, y_true):
        return torch.mean(torch.pow((y_pred - y_true), 2) * self.w)



def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features,size=output_len,align_corners=True,mode='linear')
    return output_features.transpose(1, 2)

def bs_handler(path,calibration=True):
    df = pd.read_csv(path)
    out_array = df[MOUTH_BS].to_numpy()
    if calibration:
        out_array -= out_array[0]
    out_array[out_array<0] = 0
    return out_array

def train_val_test_index(n=1000, train=0.8, validation=0.1, test=0.1):
    random.seed(2023)
    ind = np.arange(n)
    random.shuffle(ind)
    a = int(n*train)
    b = int(n*(train+validation))
    assert(train+validation+test==1 and n>2)
    return set(ind[:a]), set(ind[a:b]), set(ind[b:])


def np_to_csv(x, calibration):
    x=np.squeeze(x)
    if calibration:
        x -= x[0]
        x[x<0]=0
    pre_x = np.ones((x.shape[0],2))*31
    px = pd.DataFrame(np.concatenate([pre_x,x],axis=1),columns=["Timecode", "BlendShapeCount"]+MOUTH_BS)
    for i in range(x.shape[0]):
        h = int(i//216000)
        m = int(i//3600)-h*60
        s = int(i//60)-m*60
        f = int(i%60)
        timecode = '%02d:%02d:%02d:%02d.%d' % (h,m,s,f,i)
        px.iloc[i,0] = timecode
    return px

def exponential_smoothing(series, alpha):
    result = np.array([series[0]])
    n = np.array(series[1:])
    n_prev = np.array(series[:-1])
    result1 = alpha*n+(1-alpha)*n_prev
    return np.concatenate((result, result1))