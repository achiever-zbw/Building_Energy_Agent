import torch
import torch.nn as nn

class MainModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super().__init__()
        
        # 1. 一维卷积：捕捉局部爬坡/下降趋势（例如相邻时刻的导数关系）
        # padding=1 保持序列长度不变
        self.cnn = nn.Conv1d(
            in_channels=input_size, 
            out_channels=hidden_size, 
            kernel_size=3, 
            padding=1
        )
        self.relu = nn.ReLU()
        
        # 2. LSTM 层：接在 CNN 之后，捕捉长期周期性规律
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        
        # 3. 增强型 MLP 输出头
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, output_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        
        # CNN 需要的输入格式为 (batch, channels, seq_len)
        x_cnn = x.transpose(1, 2)
        x_cnn = self.relu(self.cnn(x_cnn))
        # 恢复回 (batch, seq_len, hidden_size) 以输入 LSTM
        x_lstm = x_cnn.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x_lstm)  # (batch, seq_len, hidden_size)
        
        # 取 LSTM 在序列最后一步（-1）的输出状态，它包含了所有历史与局部趋势记忆
        context = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        context = self.dropout(context)
        return self.fc(context)