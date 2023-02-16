import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self,args, input_size=1024, hidden_layer_size=64, output_size=31, n_head=2, max_seq_len=120):
        super().__init__()
        self.input_size=input_size
        self.out_size=output_size
        self.hidden_layer_size = hidden_layer_size
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_layer_size, nhead=n_head, dim_feedforward=2*self.hidden_layer_size, batch_first=True) 
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)
        self.audio_feature_map = nn.Linear(input_size, self.hidden_layer_size)

        self.PPE = PeriodicPositionalEncoding(self.hidden_layer_size)

        self.biased_mask = init_biased_mask(n_head = n_head, max_seq_len = max_seq_len, period=10)
        self.vertice_map_r = nn.Linear(self.hidden_layer_size, self.out_size)
        self.vertice_map = nn.Linear(self.out_size, self.hidden_layer_size)

        self.obj_vector = nn.Linear(5, self.hidden_layer_size, bias=False)
        self.device = args.device

        self.vertice_emb = None
        self.style_emb = None
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)
    
    def forward(self, audio):
        # audio 1, length, 1024
        # bs    1, length, 31
        hidden_states = self.audio_feature_map(audio)
        audio_len = audio.shape[1]
        one_hot = torch.Tensor([[1,0,0,0,0]]).to(device=self.device)
        obj_embedding = self.obj_vector(one_hot)
        frame_num = audio.shape[1]
        for i in range(frame_num):
            if self.vertice_emb is None:
                self.vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                self.style_emb = self.vertice_emb
                vertice_input = self.PPE(self.style_emb)
            else:
                vertice_input = self.PPE(self.vertice_emb)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask, memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
            new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
            new_output = new_output + self.style_emb
            self.vertice_emb = torch.cat((self.vertice_emb, new_output), 1)
            del tgt_mask
            del memory_mask
        self.vertice_emb = self.vertice_emb[:,-1,:].unsqueeze(1)
        return vertice_out
    
    def reset_hidden_cell(self):
        self.vertice_emb = None
        self.style_emb = None



"""
a = Faceformer()
audio = torch.randn(1,6,1024)
bs = torch.randn(1,6,31)
a(audio)"""