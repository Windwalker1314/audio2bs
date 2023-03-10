import torch
import torch.nn as nn
from model.modules.embedding import PositionalEmbedding, ConvEmbedding
from model.modules.attention import ProbAttention, FullAttention, AttentionLayer
from model.modules.encoder import Encoder, EncoderLayer, ConvLayer
from model.modules.decoder import Decoder, DecoderLayer

class Informer(nn.Module):
    def __init__(self, args, output_attention = False, distil=False, mix=True):
        super(Informer, self).__init__()
        self.output_attention = output_attention
        # Init hyperparameters
        self.attn = args.attention_type
        self.max_seq_length = args.max_seq_length
        self.c_out = args.c_out
        self.device= args.device
        self.d_model = args.d_model
        self.enc_in = args.enc_in
        self.dec_in = args.dec_in
        self.factor = args.factor
        self.dropout = args.dropout_rate
        self.n_heads = args.n_heads
        self.e_layers = args.n_encoder_layers
        self.d_layers = args.n_decoder_layers
        self.d_ff = args.d_feedforward
        self.activation = args.activation
        # Encoding
        self.enc_embedding = ConvEmbedding(self.enc_in, d_model=self.d_model)
        self.dec_embedding = ConvEmbedding(self.dec_in, d_model=self.d_model)
        self.enc_pe = PositionalEmbedding(d_model=self.d_model, max_len=self.max_seq_length)
        self.dec_pe = PositionalEmbedding(d_model=self.d_model, max_len=self.max_seq_length)

        # Attention
        Attn = ProbAttention if self.attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, self.factor, attention_dropout=self.dropout, output_attention=output_attention), 
                                        self.d_model, self.n_heads, mix=False),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            [
                ConvLayer(
                    self.d_model
                ) for l in range(self.e_layers-1)
            ] if distil else None,
            norm_layer=nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, self.factor, attention_dropout=self.dropout, output_attention=False), 
                                self.d_model, self.n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False), 
                                    self.d_model, self.n_heads, mix=False),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
        norm_layer=nn.LayerNorm(self.d_model)
        )
        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)
        self.x_enc_memory = None
        self.x_dec_memory = None
        
    def forward(self, x, y=None, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        pred_length = x.shape[1]
        # process encoder memory and input
        if self.x_enc_memory is None:
            self.x_enc_memory = x
        else:
            self.x_enc_memory = torch.cat((self.x_enc_memory, x), 1)           # cat(meomry_wav, current_wav)
            if self.x_enc_memory.shape[1]>self.max_seq_length:
                self.x_enc_memory = self.x_enc_memory[:,-self.max_seq_length:,:]
        # process decoder memory and input
        x_pad = torch.zeros((x.shape[0], pred_length, self.c_out)).to(self.device)  # 1, 60, 31
        if self.x_dec_memory is None:
            self.x_dec_memory = x_pad
        else:
            self.x_dec_memory = torch.cat((self.x_dec_memory, x_pad),1)
            if self.x_dec_memory.shape[1]>self.max_seq_length:
                self.x_dec_memory=self.x_dec_memory[:,-self.max_seq_length:,:]  # cat(memory_pred, zeros)
            
        enc_out = self.enc_embedding(self.x_enc_memory) + self.enc_pe(self.x_enc_memory)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(self.x_dec_memory)+self.enc_pe(self.x_dec_memory)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        output = dec_out[:,-pred_length:,:]
        self.x_dec_memory[:,-pred_length:,:] = output.detach().clone()
        

        del x_pad
        del dec_out
        del enc_out
        if self.output_attention:
            return output, attns
        return output # [B, L, D]
    
    def reset_hidden_cell(self):
        if self.x_dec_memory is not None:
            del self.x_dec_memory
        if self.x_enc_memory is not None:
            del self.x_enc_memory
        torch.cuda.empty_cache()
        self.x_enc_memory = None
        self.x_dec_memory = None