import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size=args.input_size
        self.out_size=args.output_size
        self.hidden_layer_size = args.hidden_layer_size
        self.lstm = nn.LSTM(
            self.input_size, 
            self.hidden_layer_size, 
            num_layers = args.n_layers, 
            dropout = args.dropout, 
            bidirectional=args.bidirectional, 
            batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_layer_size*2, self.output_size),
            nn.ReLU(True)
        )
        self.hidden_cell = None

    def forward(self, x):
        if self.hidden_cell is None:
            lstm_out, self.hidden_cell = self.lstm(x)
        else:
            lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        predictions = self.linear(lstm_out)
        return predictions
    
    def reset_hidden_cell(self):
        self.hidden_cell = None