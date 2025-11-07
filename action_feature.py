import torch
import torch.nn as nn

# 假设你的动作序列 shape = (64,6)
def action_sequence_generator():
    # 这里我们生成一个随机的动作序列作为示
    sequence = sequence.unsqueeze(1)  # shape -> (64, 1, 6)

    # 定义 LSTM
    input_size = 6      # 每帧6个电机角度
    hidden_size = 64    # 输出特征向量维度
    num_layers = 1      # LSTM 层数

    lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    # 前向传播
    # output: 所有时间步的隐藏状态, shape = (seq_len, batch, hidden_size)
    # (h_n, c_n): 最后时间步的隐藏状态和细胞状态, shape = (num_layers, batch, hidden_size)
    output, (h_n, c_n) = lstm(sequence)

    # 取最后一帧隐藏状态作为特征向量
    feature_vector = torch.cat([h_n[-1,0,:], c_n[-1,0,:]], dim=-1)  # shape -> (hidden_size,),h_n的shape是(num_layers, batch, hidden_size)
    # 实际上 h_n 本身只存了每层的最后一帧隐藏状态，所以我们用 h_n[-1] 就得到了最后一层的最后一帧隐藏状态。
    print("动作序列特征向量 shape:", feature_vector.shape)
    print(feature_vector)   
    # 添加 batch 维度，因为 PyTorch LSTM 需要 shape = (seq_len, batch, input_size)
    return feature_vector
    
