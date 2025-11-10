import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.policies.customACT.configuration_customACT import ACTConfig
from lerobot.policies.customACT.history_obs_state.configuration_history_obs import HistoryConv1dConfig

class CausalConv1d(nn.Module): # CausalConv：因果卷积，只会padding前面一边，未来信息范围不会padding
    # ref: https://zhuanlan.zhihu.com/p/552216156
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs ): 
        super().__init__()
        self.padding = (kernel_size -1) * dilation # 记录感受野大小，以便在 forward 时用
        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size , stride=1,
            padding=0, dilation=dilation, **kwargs # 注意这里padding是0，因为后面F.pad才是真正的padding
        )

    def forward(self, x): 
        x = F.pad(x, (self.padding , 0))
        conv1d_out = self.conv1d(x)
        return conv1d_out


class WeightedSegmentPooling(nn.Module):
    """
    ### 分段权重的池化层
    #### Args:
        num_segments (int): 分段数
        alpha (float): 控制非均匀分段的参数，aplha越大，
        decay (str): 'exponential' 或 'linear' 或 None，控制段间加权方式。
    """
    def __init__(self, num_segments=4, alpha=0.5, decay='exponential'):
        super().__init__()
        self.num_segments = num_segments # 分段数
        self.alpha = alpha # 控制非均匀分段的参数
        self.decay = decay # 'exponential' 或 'linear' 或 None

    def _make_boundaries(self, T):
        diff = self.alpha
        num_segments = self.num_segments
        if not (0 <= diff <= 1):
            raise ValueError("diff must be in [0, 1]")
        if T < num_segments:
            raise ValueError("T must >= num_segments")

        power = 1.0 - diff  # diff=0 → power=1（均匀）；diff=1 → power=0（极度前倾）
        cuts = [0]
        for i in range(1, num_segments):
            ratio = (i / num_segments) ** (1.0 / (power + 1e-9))
            pos = int(round(ratio * T))
            cuts.append(max(pos, cuts[-1] + 1))  # 至少比前一个大1
        cuts.append(T) # [0,1,8,32]
        cuts = cuts[::-1] # [32,8,1,0]

        return [(T-cuts[i], T-cuts[i+1]) for i in range(num_segments)]

    def forward(self, x):
        B, C, T = x.shape # [B, C, T]
        boundaries = self._make_boundaries(T)

        segment_feats = []
        prev = 0
        # 先每段单独池化再加权，相当于每步的权重是w/L，否则段落的长度会影响每段的权值
        for boundary in boundaries:
                print(boundary)
                seg = x[:, :, boundary[0]:boundary[1]]  # [B, C, L] 取时间轴的其中一段
                seg_mean = seg.mean(dim=-1)  # [B, C] 先对段做简单平均池化（得到每段的平均值）
                segment_feats.append(seg_mean)
        seg_feats = torch.stack(segment_feats, dim=-1)  # [B, C, num_segments]

        # 段间权值
        if self.decay == 'exponential': # 指数递增 比如[0.14, 0.37, 1.0]
            weights = torch.exp(torch.linspace(-2.0, 0.0, self.num_segments, device=x.device))
        elif self.decay == 'linear': # 线性递增 比如[0.1, 0.3, 0.5, 0.7, 1.0]
            weights = torch.linspace(1/self.num_segments, 1.0, self.num_segments, device=x.device)
        else: # 平权 [1, 1, 1, ...]
            weights = torch.ones(self.num_segments, device=x.device)
        weights = weights / weights.sum() # 归一化，保证加权后结果仍是加权平均（而不是加权求和）

        out = torch.sum(seg_feats * weights[None, None, :], dim=-1)  # [B, C]（其中 weights[None, None, :]是扩展维度，使其到 [B, C, num_segments]）
        return out

class HistoryConv1dEmbedding(nn.Module): # 卷积特征
    def __init__(self, config: ACTConfig, modeling_config: HistoryConv1dConfig):
        super().__init__()
        self.history_segment_num = modeling_config.history_segment_num
        self.history_segment_alpha = modeling_config.history_segment_alpha
        self.history_segment_decay = modeling_config.history_segment_decay
        
        self.motion_encoder = nn.Sequential( # 三层卷积提取特征
            # Layer 1
            CausalConv1d(in_channels=config.robot_state_feature.shape[0], out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # Layer 2
            CausalConv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # Layer 3
            CausalConv1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            # nn.AdaptiveAvgPool1d(1),  # [B, 256, T] => [B, 256, 1]
            # nn.Flatten(start_dim=1),   # [B, 256]
            # nn.Linear(256, config.dim_model) # [B, 256] => [B, dim_model]
        )
        self.segment_pool = WeightedSegmentPooling(
            num_segments=self.history_segment_num,
            alpha=self.history_segment_alpha,
            decay=self.history_segment_decay
        )

        self.proj = nn.Sequential( # input = # [B, 256]
            nn.Flatten(start_dim=1), # [B, 256]
            nn.Linear(256, config.dim_model) # [B, dim_model]
        )

    def forward(self, x): # input = [B, T, state_dim]
        x = x.permute(0, 2, 1)  # [B, state_dim, T]
        feat = self.motion_encoder(x)     # [B, 256, T]
        pooled = self.segment_pool(feat)  # [B, 256]
        out = self.proj(pooled)           # [B, dim_model]
        return out

