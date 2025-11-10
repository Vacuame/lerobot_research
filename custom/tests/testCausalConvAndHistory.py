
import torch
from lerobot.policies.customACT.history_obs_state.embedding_conv1d_history_obs import CausalConv1d


class CausalConv1d_QWEN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation #padding的作用是使输入长度=输出
        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation, **kwargs
        )
    def forward(self, x):
        x = self.conv(x) # x: [B, C, T]  (注意: Conv1d 要求通道在第二维)
        # 切掉右侧 padding (移除padding生成的未来信息)
        x = x[:, :, :-self.padding] if self.padding > 0 else x
        return x


def testCov():
    print('Testing CausalConv1d...')
    C, T = 6, 10 # 6电机 10时间步 电机是channel
    # x = torch.randn(1, C, T) 
    x = torch.tensor([[[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],
                      [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],
                      [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],
                      [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],
                      [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],
                      [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]]] )
    # print(x)
    causal_conv = CausalConv1d(C, 64, kernel_size=3)
    causal_conv_QWEN = CausalConv1d_QWEN(C, 64, kernel_size=3)
    y = causal_conv(x)
    yq = causal_conv_QWEN(x)

    print(y.shape)
    print(y[0][1])
    print(yq.shape)
    print(yq[0][1])


def testHistoryObsStateEmbedding():
    print('Testing HistoryObsStateEmbedding...')
    from lerobot.policies.customACT.modeling_customACT import HistoryObsStateEmbedding
    from lerobot.policies.customACT.configuration_customACT import ACTConfig
    C, T = 6, 10 # 6电机 10时间步 电机是channel
    x = torch.tensor([[[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],
                      [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],
                      [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],
                      [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],
                      [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],
                      [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]]] )
    config = ACTConfig()
    # config.robot_state_feature.shape[0] = C
    history_obs_state_embedding = HistoryObsStateEmbedding(config)
    y = history_obs_state_embedding(x)
    print(y.shape)  # Expected: [B, dim_model]

def split_segments(len, num, diff):
    if not (0 <= diff <= 1):
        raise ValueError("diff must be in [0, 1]")
    if len < num:
        raise ValueError("len must be >= num")
    if num == 1:
        return [(0, len)]

    power = 1.0 - diff  # diff=0 → power=1（均匀）；diff=1 → power=0（极度前倾）
    cuts = [0]
    for i in range(1, num):
        ratio = (i / num) ** (1.0 / (power + 1e-9))
        pos = int(round(ratio * len))
        cuts.append(max(pos, cuts[-1] + 1))  # 至少比前一个大1
    cuts.append(len) # [0,1,8,32]
    cuts = cuts[::-1] # [32,8,1,0]
    
    return [(len-cuts[i], len-cuts[i+1],cuts[i]-cuts[i+1]) for i in range(num)]

def 测试分组(len,num,diff):
    a = split_segments(len, num, diff)
    print('x1,x2,length:')
    print(a)

    weights1 = torch.exp(torch.linspace(-2.0, 0.0, num))
    weights2 = torch.linspace(1/num, 1.0, num)
    print(f'Exponential weights: {weights1}   Linear weights: {weights2}')


if __name__ == "__main__":
    测试分组(64,3, 0.5)
    