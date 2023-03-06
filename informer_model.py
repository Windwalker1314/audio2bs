from informer.encoder import Encoder, EncoderLayer, ConvLayer
from informer.decoder import Decoder, DecoderLayer
from informer.attention import FullAttention, ProbAttention, AttentionLayer
import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class ConvEmbedding(nn.Module):
    def __init__(self, c_in=1024, d_model=512):
        super(ConvEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='replicate')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class Informer(nn.Module):
    def __init__(self, args, max_seq_length=1024,
                factor=5, d_model=256, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.05, attn='prob', activation='gelu', 
                output_attention = False, distil=False, mix=True):
        super(Informer, self).__init__()
        self.attn = attn
        self.output_attention = output_attention
        self.max_seq_length = max_seq_length
        self.c_out = args.c_out
        self.device= args.device
        # Encoding
        self.enc_embedding = ConvEmbedding(args.enc_in, d_model=d_model)
        self.dec_embedding = ConvEmbedding(args.dec_in, d_model=d_model)
        self.enc_pe = PositionalEmbedding(d_model=d_model)
        self.dec_pe = PositionalEmbedding(d_model=d_model)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
        norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, self.c_out, bias=True)
        self.x_enc_memory = None
        self.x_dec_memory = None
        
    def forward(self, x, y=None, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        pred_length = x.shape[1]
        if self.x_enc_memory is None:
            self.x_enc_memory = x
        else:
            self.x_enc_memory = torch.cat((self.x_enc_memory, x), 1)
            if self.x_enc_memory.shape[1]>self.max_seq_length:
                self.x_enc_memory = self.x_enc_memory[:,-self.max_seq_length:,:]
        x_pad = torch.zeros((x.shape[0], pred_length, self.c_out)).to(self.device)  # 1, 60, 31
        if self.x_dec_memory is not None:
            x_dec = torch.cat((self.x_dec_memory, x_pad),1)
        else:
            x_dec = x_pad

        enc_out = self.enc_embedding(self.x_enc_memory) +self.enc_pe(self.x_enc_memory)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec) + self.dec_pe(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)

        if self.x_dec_memory is None:
            self.x_dec_memory = dec_out
        else:
            if y is not None:
                self.x_dec_memory = torch.cat((self.x_dec_memory, y), 1)
            else:
                self.x_dec_memory = torch.cat((self.x_dec_memory, dec_out),1)
            if self.x_dec_memory.shape[1]>self.max_seq_length:
                self.x_dec_memory=self.x_dec_memory[:,-self.max_seq_length:,:]
        if self.output_attention:
            return dec_out[:,-pred_length:,:], attns
        else:
            return dec_out[:,-pred_length:,:] # [B, L, D]
    
    def reset_hidden_cell(self):
        self.x_enc_memory = None
        self.x_dec_memory = None