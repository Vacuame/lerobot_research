import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.customACT.configuration_customACT import ACTConfig
from lerobot.policies.customACT.history_obs_state.configuration_history_obs import HistoryConv1dConfig


class CausalConv1d(nn.Module):
    """一维因果卷积。

    输入形状是 [B, C, T]，其中 T 是时间维度。普通 Conv1d 如果直接 padding，
    卷积窗口可能会看到当前位置右侧的未来帧；这里手动只在左侧 padding，
    保证第 t 帧的输出只依赖 t 以及 t 之前的历史。
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        """初始化因果卷积层。

        Args:
            in_channels: 输入通道数，比如状态维度或上一层特征维度。
            out_channels: 输出通道数。
            kernel_size: 卷积核大小。
            dilation: 空洞卷积间隔，用来扩大历史感受野。
            **kwargs: 传给 nn.Conv1d 的其他参数。
        """
        super().__init__()
        # 只在时间轴左侧补这么多 0，让输出长度仍然等于输入长度。
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x):
        """执行因果卷积。

        Args:
            x: [B, C, T]，B 是 batch，C 是通道，T 是历史长度。

        Returns:
            [B, out_channels, T]，时间长度保持不变。
        """
        x = F.pad(x, (self.padding, 0))
        return self.conv1d(x)


class SegmentStatsPooling(nn.Module):
    """把完整历史序列压缩成固定数量的 history tokens。

    Conv1d 会输出每一帧的特征 [B, C, T]。如果直接把 T 个 token 都丢给 ACT，
    token 数太多，训练和推理成本都会上升。这里把历史按时间分成 num_segments 段，
    每段提取 mean / max / last 三种统计量：
    - mean 表示这一段的平均状态特征；
    - max 保留这一段里最强的激活；
    - last 保留这一段末尾的状态，更贴近当前时刻。

    最终输出 [B, num_segments, C * 3]，每个 segment 对应一个 history token。
    """

    def __init__(self, num_segments=4, alpha=0.3, decay=None):
        """初始化分段池化层。

        Args:
            num_segments: 把历史序列切成多少段，也就是输出多少个 history tokens。
            alpha: 控制非均匀切分程度。0 接近均匀切分，越大越偏向保留近端历史细节。
            decay: 预留参数；当前实现不再用固定权重或可学习 scalar 给段落加权。
        """
        super().__init__()
        self.num_segments = num_segments
        self.alpha = alpha
        self.decay = decay

    def _make_boundaries(self, T):
        """根据历史长度 T 计算每个 segment 的 [start, end) 区间。

        返回的区间从较早历史到较近历史排列。alpha 会影响切分位置：
        - alpha 越小，每段长度越接近；
        - alpha 越大，越倾向于让靠近当前时刻的 segment 更短、更细。

        Args:
            T: 历史序列长度。

        Returns:
            list[tuple[int, int]]，每个元素是一个左闭右开区间。
        """
        diff = self.alpha
        num_segments = self.num_segments
        if not (0 <= diff <= 1):
            raise ValueError("diff must be in [0, 1]")
        if T < num_segments:
            raise ValueError("T must >= num_segments")

        power = 1.0 - diff
        cuts = [0]
        for i in range(1, num_segments):
            ratio = (i / num_segments) ** (1.0 / (power + 1e-9))
            pos = int(round(ratio * T))
            cuts.append(max(pos, cuts[-1] + 1))
        cuts.append(T)
        cuts = cuts[::-1]
        return [(T - cuts[i], T - cuts[i + 1]) for i in range(num_segments)]

    def forward(self, x):
        """对 Conv1d 后的历史特征做分段统计池化。

        Args:
            x: [B, C, T]，Conv1d 编码后的逐帧历史特征。

        Returns:
            [B, num_segments, C * 3]，每段拼接 mean/max/last 三种统计特征。
        """
        _, _, T = x.shape
        boundaries = self._make_boundaries(T)

        segment_feats = []
        for start, end in boundaries:
            seg = x[:, :, start:end]
            # mean 看整体趋势，max 保留显著响应，last 保留段尾的近时刻信息。
            seg_mean = seg.mean(dim=-1)
            seg_max = seg.max(dim=-1).values
            seg_last = seg[:, :, -1]
            segment_feats.append(torch.cat([seg_mean, seg_max, seg_last], dim=-1))

        return torch.stack(segment_feats, dim=1)


class HistoryConv1dEmbedding(nn.Module):
    """历史关节状态序列的 Conv1d embedding 模块。

    输入是 history_obs_states，形状 [B, T, state_dim]。这里没有直接使用绝对状态，
    而是构造两类更适合表达运动趋势的特征：
    - rel_state: 每一帧相对当前帧的偏移，减少和当前 observation.state token 的重复；
    - velocity: 相邻帧差分，显式告诉模型过去一段时间的运动方向和速度趋势。

    然后用多层因果 Conv1d 编码时间信息，再用 SegmentStatsPooling 压缩成固定数量
    的 history tokens，最后投影到 ACT transformer 的 dim_model。
    """

    def __init__(self, config: ACTConfig, modeling_config: HistoryConv1dConfig):
        """初始化历史状态编码器。

        Args:
            config: CustomACT 的主配置，提供 state_dim 和 dim_model。
            modeling_config: 历史模块配置，提供 segment 数量和切分参数。
        """
        super().__init__()
        self.history_segment_num = modeling_config.history_segment_num
        self.history_segment_alpha = modeling_config.history_segment_alpha
        self.history_segment_decay = modeling_config.history_segment_decay

        state_dim = config.robot_state_feature.shape[0]
        # 输入通道是 state_dim * 2，因为 forward 里会拼接 rel_state 和 velocity。
        # dilation=1/2/4 逐层扩大历史感受野，让输出既能看短期变化，也能看更长趋势。
        self.history_encoder = nn.Sequential(
            CausalConv1d(in_channels=state_dim * 2, out_channels=64, kernel_size=3, dilation=1),
            nn.ReLU(),
            # GroupNorm 比 BatchNorm 更适合小 batch 机器人训练，统计量不依赖 batch 大小。
            nn.GroupNorm(8, 64),
            CausalConv1d(in_channels=64, out_channels=128, kernel_size=3, dilation=2),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            CausalConv1d(in_channels=128, out_channels=256, kernel_size=3, dilation=4),
            nn.ReLU(),
        )
        # 把 T 帧 Conv 特征压成固定的 num_segments 个 token。
        self.segment_pool = SegmentStatsPooling(
            num_segments=self.history_segment_num,
            alpha=self.history_segment_alpha,
            decay=self.history_segment_decay,
        )
        # 每个 segment 的特征是 256 通道的 mean/max/last 拼接，所以输入维度是 256 * 3。
        # LayerNorm 用来让 history token 的尺度更接近其他 transformer token。
        self.proj = nn.Sequential(
            nn.Linear(256 * 3, config.dim_model),
            nn.LayerNorm(config.dim_model),
        )

    def forward(self, x):
        """把历史状态序列编码为 ACT encoder 可用的 history tokens。

        Args:
            x: [B, T, state_dim]，按时间从旧到新排列，最后一帧是当前状态。

        Returns:
            [B, num_segments, dim_model]，可直接作为多个 token 拼进 ACT encoder。
        """
        # 相对当前状态：强调“过去离现在有多远”，弱化绝对关节角的重复信息。
        rel_state = x - x[:, -1:, :]
        # 一阶差分：显式建模速度/方向。第一帧没有前一帧，所以保持为 0。
        velocity = torch.zeros_like(x)
        velocity[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        # 拼接后每个时间步的通道数从 state_dim 变为 state_dim * 2。
        x = torch.cat([rel_state, velocity], dim=-1)

        # Conv1d 要求通道在第 2 维，所以从 [B, T, C] 转成 [B, C, T]。
        x = x.permute(0, 2, 1)
        feat = self.history_encoder(x)
        pooled = self.segment_pool(feat)
        return self.proj(pooled)
