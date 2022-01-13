import torch.nn as nn

class LSTMAnomalyDetector(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,dropout,bidirectional):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,batch_first=True,dropout=dropout,bidirectional=bidirectional,num_layers=2)
        self.relu = nn.ReLU()
        self.dense = nn.Linear(2*hidden_size if bidirectional else hidden_size, output_size)
        
    
    def forward(self, input):
        
        lstm_output,_ = self.lstm(input)
        relu_output = self.relu(lstm_output[:,-1,:])
        
        
        return self.dense(relu_output)
        