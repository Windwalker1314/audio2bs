import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor
from transformers import HubertModel
import torch.nn.functional as F
import numpy as np
import pandas as pd
import resampy
import time

class LSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_layer_size=128, output_size=31):
        super().__init__()
        self.input_size=input_size
        self.out_size=output_size
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=2, batch_first= True,  dropout=0.5, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_layer_size*2, output_size),
            nn.ReLU(True)
        )
        self.hidden_cell = None


    def forward(self, x):
        if self.hidden_cell is None:
            lstm_out, self.hidden_cell = self.lstm(x)
        else:
            lstm_out, self.hidden_cell = self.lstm(x,self.hidden_cell)
        predictions = self.linear(lstm_out)
        return predictions
    
    def reset_hidden_cell(self):
        self.hidden_cell = None

class Audio2BS():
    def __init__(self, base_model_path, model_path, device, 
                output_fps = 60, base_model_type = "Hubert", 
                audio_section_length=1) -> None:
        self.device = device
        self.base_model_type = base_model_type
        self.audio_section_length = audio_section_length
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(base_model_path)
        self.model = self.load_model(model_path)
        self.base_model, self.fps = self.load_base_model(base_model_path)
        self.output_fps = output_fps
        self.MOUTH_BS = ['JawForward', 'JawRight', 'JawLeft', 'JawOpen',
                        'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthRight', 'MouthLeft',
                        'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft',
                        'MouthFrownRight', 'MouthDimpleLeft', 'MouthDimpleRight',
                        'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower',
                        'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper',
                        'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft',
                        'MouthLowerDownRight', 'MouthUpperUpLeft', 'MouthUpperUpRight',
                        'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight','TongueOut']
        self.model.reset_hidden_cell()
        torch.cuda.empty_cache()
                        
    def load_model(self, model_path:str):
        if model_path.endswith(".pth"):
            model = LSTM()
            checkpoint_path = model_path
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif model_path.endswith(".pt"):
            model = torch.load(model_path)
        
        return model.to(self.device).eval()
    
    def load_base_model(self, model_path, model_type="Hubert"):
        if model_type=="Hubert":
            model = HubertModel.from_pretrained(model_path)
            model_fps = 49
        else:
            raise NameError
        
        return model.to(self.device).eval(), model_fps
    
    def linear_interpolation(self, features, input_fps, output_fps, output_len=None):
        features = features.transpose(1, 2)
        seq_len = features.shape[2] / float(input_fps)
        if output_len is None:
            output_len = int(seq_len * output_fps)
        output_features = F.interpolate(features,size=output_len,align_corners=True,mode='linear')
        return output_features.transpose(1, 2)

    def inference(self, audio, rate):
        if type(audio[0])==np.int16:
            audio = np.array(audio)/32768
        elif type(audio[0])==np.uint8:
            audio = np.array(audio).astype(np.int8)/128.0
        elif type(audio[0])==list:
            audio = np.array(audio)[:, 0]
        elif type(audio[0]==float):
            audio = np.array(audio)
        else:
            raise TypeError
        
        audio = resampy.resample(audio.astype(float),rate,16000)
        audio = self.processor(audio, return_tensors="pt", sampling_rate=16000).input_values
        if audio.shape[1]<32000:
            x = torch.FloatTensor(audio).to(dtype=torch.float32, device=self.device)
            with torch.no_grad():
                last_h_state = self.base_model(x).last_hidden_state
            x = self.linear_interpolation(last_h_state, input_fps=self.fps, output_fps=60)
            with torch.no_grad():
                output = self.model(x)
                output = output.detach().cpu().numpy()
            torch.cuda.empty_cache()
            return output
        chunks = torch.split(audio, self.audio_section_length*rate, dim=1)
        output = []
        for chunk in chunks:
            x = torch.FloatTensor(chunk).to(dtype=torch.float32, device=self.device)
            if chunk.shape[1]<640:
                break
            with torch.no_grad():
                last_h_state = self.base_model(x).last_hidden_state
            x = self.linear_interpolation(last_h_state, input_fps=self.fps, output_fps=60)
            with torch.no_grad():
                x = self.model(x).detach().cpu().numpy()

            output.append(x)
            del x
            del chunk
            del last_h_state
            torch.cuda.empty_cache()
        if len(output)>0:
            x = np.concatenate(output,axis=1)
            return x
        return output
    
    def np_to_csv(self, x, calibration):
        x=np.squeeze(x)
        if calibration:
            x -= x[0]
            x[x<0]=0
        pre_x = np.ones((x.shape[0],2))*31
        px = pd.DataFrame(np.concatenate([pre_x,x],axis=1),columns=["Timecode", "BlendShapeCount"]+ self.MOUTH_BS)
        for i in range(x.shape[0]):
            h = int(i//216000)
            m = int(i//3600)-h*60
            s = int(i//60)-m*60
            f = int(i%60)
            timecode = '%02d:%02d:%02d:%02d.%d' % (h,m,s,f,i)
            px.iloc[i,0] = timecode
        return px

    def reset_hidden_state(self):
        self.model.reset_hidden_cell()