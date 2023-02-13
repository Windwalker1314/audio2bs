import torch.nn as nn

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