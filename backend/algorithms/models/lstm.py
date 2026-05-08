import torch
import torch.nn as nn

class LSTM(nn.Module) :
    """ LSTM 模型 """
    def __init__(self , input_size , hidden_size , num_layers , output_size  , dropout_rate ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size , hidden_size , num_layers , batch_first = True , dropout = dropout_rate 
        )

        # 简化的输出层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self , x):
        # LSTM输出
        out , _ = self.lstm(x)
        # 取最后一个时间步
        out = out[:,-1,:]
        # Dropout和线性输出
        out = self.dropout(out)
        out = self.fc(out)
        return out
    