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


class WeightedSegmentPooling(nn.Module): # 自适应非均匀分段 + 段内指数加权平均
    def __init__(self, num_segments=4, alpha=0.5, decay='exponential'):
        super().__init__()
        self.num_segments = num_segments # 分段数
        self.alpha = alpha # 控制非均匀分段的参数
        self.decay = decay # 'exponential' 或 'linear' 或 None

    def _make_boundaries(self, T):
        # 根据帧数 T 自动生成非均匀边界
        ratios = [1 - self.alpha**(i / self.num_segments) for i in range(self.num_segments + 1)]
        indices = [int(r * (T - 1)) for r in ratios]
        boundaries = [(indices[i], indices[i+1]) for i in range(self.num_segments)]
        return boundaries

    def forward(self, x):
        B, C, T = x.shape # [B, C, T]
        boundaries = self._make_boundaries(T)

        pooled = []
        for (start, end) in boundaries:
            seg = x[:, :, start:end+1]  # [B, C, L]
            L = seg.shape[-1]

            # 段内权重
            if self.decay == 'exponential':
                weights = torch.exp(torch.linspace(-2.0, 0.0, L, device=x.device))
            elif self.decay == 'linear':
                weights = torch.linspace(0.1, 1.0, L, device=x.device)
            else:
                weights = torch.ones(L, device=x.device)

            weights = weights / weights.sum()
            seg_weighted = torch.sum(seg * weights[None, None, :], dim=-1)  # [B, C]
            pooled.append(seg_weighted)

        out = torch.stack(pooled, dim=-1)  # [B, C, num_segments]
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
            # nn.AdaptiveAvgPool1d(1),  # [B, 256, n] -> [B, 256, 1]
            # nn.Flatten(start_dim=1),   # [B, 256]
            # nn.Linear(256, config.dim_model) # [B, 256]
        )
        self.segment_pool = WeightedSegmentPooling(
            num_segments=self.history_segment_num,
            alpha=self.history_segment_alpha,
            decay=self.history_segment_decay
        )

        self.proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(256 * self.history_segment_num, config.dim_model)
        )

    def forward(self, x): # x: [B, T, state_dim]
        x = x.permute(0, 2, 1)  # [B, state_dim, T]
        feat = self.motion_encoder(x)     # [B, 256, T]
        pooled = self.segment_pool(feat)  # [B, 256, num_segments]
        out = self.proj(pooled)           # [B, dim_model]=[batch_size,512]
        return out

