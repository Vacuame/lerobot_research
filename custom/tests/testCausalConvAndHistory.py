
import torch
from lerobot.policies.customACT.modeling_customACT import CausalConv1d


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


if __name__ == "__main__":
    testHistoryObsStateEmbedding()