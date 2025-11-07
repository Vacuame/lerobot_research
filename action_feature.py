import torch
import torch.nn as nn

class ActionFeatureExtractor(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, feature_dim=512, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # 让输入支持 (batch, seq_len, input_size)
        )
        self.fc = nn.Linear(hidden_size * 2, feature_dim)

    def forward(self, sequence):
        """
        sequence: shape (batch, seq_len, input_size)
                  比如 (1, 64, 6)
        """
        output, (h_n, c_n) = self.lstm(sequence)
        # 取最后一层的最后一帧隐藏状态和细胞状态
        h_last = h_n[-1]   # shape: (batch, hidden_size)
        c_last = c_n[-1]
        feature_vector = torch.cat([h_last, c_last], dim=-1)  # (batch, hidden_size*2)
        feature_vector = self.fc(feature_vector)               # (batch, feature_dim)
        return feature_vector
