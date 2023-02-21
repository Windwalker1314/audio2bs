import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S) # T小 S大
    for i in range(min(T,S)):
        mask[-(i+1), -(i+1)] = 0
    return (mask==1).to(device=device)

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=10, max_seq_len=180):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len=600):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

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


class Faceformer(nn.Module):
    def __init__(self,args,input_size=1024, hidden_layer_size=64, output_size=31, n_head=2, max_seq_len=180):
        super().__init__()
        self.input_size=input_size
        self.out_size=output_size
        self.hidden_layer_size = hidden_layer_size
        self.max_seq_len = max_seq_len
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_layer_size, nhead=n_head, dim_feedforward=2*self.hidden_layer_size, batch_first=True) 
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)
        self.audio_feature_map = nn.Linear(input_size, self.hidden_layer_size)

        self.PPE = PositionalEncoding(self.hidden_layer_size, max_seq_len=max_seq_len)

        self.biased_mask = init_biased_mask(n_head = n_head, max_seq_len = max_seq_len, period=10)
        self.vertice_map_r = nn.Linear(self.hidden_layer_size, self.out_size)
        self.vertice_map = nn.Linear(self.out_size, self.hidden_layer_size)

        self.device = args.device

        self.last_bs = None
        self.hidden_states = None
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)
    
    def forward(self, audio,bs,teacher_forcing=0):
        # audio 1, length(n_frames), 1024
        # bs    1, length(n_frames), 31
        assert audio.shape[1]<self.max_seq_len, "Audio sequence is too long:"+str(audio.shape[1])+" Expected:"+str(self.max_seq_len)
        if self.hidden_states is None:
            self.hidden_states = self.audio_feature_map(audio)
        else:
            self.hidden_states = torch.cat((self.hidden_states,self.audio_feature_map(audio)),1)
            if self.hidden_states.shape[1]>self.max_seq_len:
                self.hidden_states = self.hidden_states[:,-self.max_seq_len:,:]

        frame_num = audio.shape[1]

        if teacher_forcing>0.99:
            if self.last_bs is None:
                self.last_bs = torch.zeros((1,1,31)).to(device=self.device)
            bs_input = torch.cat((self.last_bs[:,-1,:].unsqueeze(1), bs[:,:-1,:]), 1)  # (1,1,31), (1,L-1, 31)
            bs_input = self.vertice_map(bs_input)
            bs_input = self.PPE(bs_input)
            len_past_pred = bs_input.shape[1]
            tgt_mask = self.biased_mask[:, :len_past_pred, :len_past_pred].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, len_past_pred, self.hidden_states.shape[1])
            """bs_out_h = self.transformer_decoder(bs_input, self.hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask).detach()
            mask = torch.bernoulli(torch.full(bs_out_h.shape, teacher_forcing)).int().to(device=self.device)
            reverse_mask = torch.ones(bs_out_h.shape).int().to(device=self.device) - mask
            bs_out_h=self.PPE(bs_out_h)
            bs_input_second = bs_input * mask + bs_out_h * reverse_mask
            """
            bs_out = self.transformer_decoder(bs_input, self.hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            bs_out = self.vertice_map_r(bs_out)
            self.last_bs = bs_out
            del tgt_mask
            del memory_mask
            return bs_out
        
        for i in range(frame_num):
            if i==0:
                if self.last_bs is None:
                    self.last_bs = torch.zeros((1,1,31)).to(device=self.device)
                bs_input = self.vertice_map(self.last_bs)
                bs_input = self.PPE(bs_input)  # 64
            else:
                bs_input = self.PPE(bs_input)
            if bs_input.shape[1]>frame_num:
                bs_input=bs_input[:,-frame_num:,:]
            len_past_pred = bs_input.shape[1]
            tgt_mask = self.biased_mask[:, :len_past_pred, :len_past_pred].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, len_past_pred, self.hidden_states.shape[1])
            bs_out = self.transformer_decoder(bs_input, self.hidden_states, tgt_mask, memory_mask)
            bs_out = self.vertice_map_r(bs_out)  # 31
            new_output = self.vertice_map(bs_out[:,-1,:]).unsqueeze(1) # b,1,64
            is_teacher = random.random() < teacher_forcing
            if is_teacher:
                ground_truth = self.vertice_map(bs[:,i,:]).unsqueeze(1)
                bs_input = torch.cat((bs_input, ground_truth), 1)
            else:
                bs_input = torch.cat((bs_input, new_output), 1)
            del tgt_mask
            del memory_mask
        self.last_bs = bs_out
        return bs_out
    
    def reset_hidden_cell(self):
        self.last_bs = None
        self.hidden_states = None



"""
a = Faceformer()
audio = torch.randn(1,60,1024)
audio1 = torch.randn(1,31,1024)
bs = torch.randn(1,60,31)
print(a(audio,bs).shape)
print(a(audio1).shape)"""



class Transformer(nn.Module):
    def __init__(self,input_size=1024, hidden_layer_size=128, output_size=31, n_head=4, max_seq_len=300):
        super().__init__()
        self.input_size=input_size
        self.out_size=output_size
        self.hidden_layer_size = hidden_layer_size
        self.max_seq_len = max_seq_len

        self.ffn = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.Dropout(p=0.2),
            nn.ReLU(True)
        )

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_layer_size, nhead=n_head, batch_first=True, dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.PE = PositionalEncoding(self.hidden_layer_size, max_seq_len=max_seq_len)

        self.linear = nn.Sequential(
            nn.Linear(hidden_layer_size, output_size),
            nn.ReLU(True)
        )

        self.memory = None

    def forward(self,audio):
        # audio 1, length(n_frames), 1024
        if self.memory is not None:
            self.memory = torch.cat((self.memory, audio), 1)
            if self.memory.shape[1]>self.max_seq_len:
                self.memory = self.memory[:,-self.max_seq_len:,:]
        else:
            self.memory = audio
        
        x = self.ffn(self.memory)
        x = self.PE(x)
        x = self.transformer_encoder(x)
        x = self.linear(x)
        return x[:,-audio.shape[1]:,:]
    
    def reset_hidden_cell(self):
        self.memory = None

class Conformer(nn.Module):
    def __init__(self,input_size=1024, hidden_layer_size=128, output_size=31, n_head=4, max_seq_len=300):
        super().__init__()
        self.input_size=input_size
        self.out_size=output_size
        self.hidden_layer_size = hidden_layer_size
        self.max_seq_len = max_seq_len

        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, stride=2)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_layer_size, nhead=n_head, batch_first=True, dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.PE = PositionalEncoding(self.hidden_layer_size, max_seq_len=max_seq_len)

        self.linear = nn.Sequential(
            nn.Linear(hidden_layer_size, output_size),
            nn.ReLU(True)
        )

        self.memory = None

    def forward(self,audio):
        # audio 1, length(n_frames), 1024
        if self.memory is not None:
            self.memory = torch.cat((self.memory, audio), 1)
            if self.memory.shape[1]>self.max_seq_len:
                self.memory = self.memory[:,-self.max_seq_len:,:]
        else:
            self.memory = audio
        
        x = self.ffn(self.memory)
        x = self.PE(x)
        x = self.transformer_encoder(x)
        x = self.linear(x)
        return x[:,-audio.shape[1]:,:]
    
    def reset_hidden_cell(self):
        self.memory = None

m = nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=3, stride=2)
a= torch.randn(1, 1024, 60)
print(m(a).shape)