import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.customACT.configuration_customACT import ACTConfig
from lerobot.policies.customACT.history_obs_state.configuration_history_obs import HistoryConv1dConfig


class CausalConv1d(nn.Module):
    """一维因果卷积。

    输入形状是 [B, C, T]，其中 T 是时间维度。这里手动只在左侧 padding，
    保证第 t 帧的输出只依赖 t 以及 t 之前的历史，不会把右侧未来帧混进来。
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


class EventAwareSegmentPooling(nn.Module):
    """把历史序列压缩成固定数量的、带事件意识的 history tokens。

    每个 segment 输出三种信息：
    - mean token: 这一段的整体运动趋势；
    - last token: 这一段末尾、最接近下一段的状态；
    - event token: 这一段里最像“事件”的时刻，例如速度/加速度突变、接触、卡住或阶段切换。

    event token 不是人工阶段标签。它由两部分共同决定：
    - learnable scorer: 从 Conv1d 特征里学习哪些时刻对动作预测重要；
    - event_prior: 由速度和加速度幅值构造的无监督运动突变先验。
    """

    def __init__(self, num_segments=4, alpha=0.3, feature_dim=256, event_prior_weight=1.0):
        """初始化事件感知分段池化层。

        Args:
            num_segments: 把历史序列切成多少段，也就是输出多少个 history tokens。
            alpha: 控制非均匀切分程度。0 接近均匀切分，越大越偏向保留近端历史细节。
            feature_dim: Conv1d 输出的通道数。
            event_prior_weight: 运动突变先验在 event logits 里的权重。
        """
        super().__init__()
        self.num_segments = num_segments
        self.alpha = alpha
        self.event_prior_weight = event_prior_weight
        self.event_scorer = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(feature_dim // 4, 1, kernel_size=1),
        )

    def _make_boundaries(self, T):
        """根据历史长度 T 计算每个 segment 的 [start, end) 区间。

        返回的区间从较早历史到较近历史排列。alpha 越大，越倾向于把靠近当前
        时刻的历史切得更细，这通常更适合机器人控制，因为最近变化更重要。
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

    def forward(self, x, event_prior=None):
        """对 Conv1d 后的历史特征做事件感知池化。

        Args:
            x: [B, C, T]，Conv1d 编码后的逐帧历史特征。
            event_prior: [B, T]，由速度和加速度构造的运动突变先验。可以为 None。

        Returns:
            [B, num_segments, C * 3]，每段拼接 mean/last/event 三种特征。
        """
        _, _, T = x.shape
        boundaries = self._make_boundaries(T)
        event_logits = self.event_scorer(x).squeeze(1)

        if event_prior is not None:
            event_prior = (event_prior - event_prior.mean(dim=1, keepdim=True)) / (
                event_prior.std(dim=1, keepdim=True) + 1e-6
            )
            event_logits = event_logits + self.event_prior_weight * event_prior

        segment_feats = []
        for start, end in boundaries:
            seg = x[:, :, start:end]
            seg_logits = event_logits[:, start:end]
            event_weight = torch.softmax(seg_logits, dim=-1)

            seg_mean = seg.mean(dim=-1)
            seg_last = seg[:, :, -1]
            seg_event = torch.sum(seg * event_weight.unsqueeze(1), dim=-1)

            segment_feats.append(torch.cat([seg_mean, seg_last, seg_event], dim=-1))

        return torch.stack(segment_feats, dim=1)


class HistoryConv1dEmbedding(nn.Module):
    """事件感知历史状态 token 模块。

    输入仍然是当前数据集已有的 history_obs_states: [B, T, state_dim]。
    为了让历史更有决策意义，这里不直接喂绝对关节角，而是构造：
    - rel_state: 历史每一帧相对当前帧的偏移，用来表示任务进度和过去姿态；
    - velocity: 一阶差分，用来表示运动方向和速度；
    - acceleration: 二阶差分，用来表示速度突变，常对应接触、卡住或阶段切换。

    这三个特征经过 dilated causal Conv1d 后，再由 EventAwareSegmentPooling
    压缩成固定数量的 history tokens，最后投影到 ACT 的 dim_model。
    """

    def __init__(self, config: ACTConfig, modeling_config: HistoryConv1dConfig):
        """初始化历史状态编码器。

        Args:
            config: CustomACT 主配置，提供 state_dim、dim_model 等模型参数。
            modeling_config: 历史模块配置，提供 segment 数量、切分参数和事件先验权重。
        """
        super().__init__()
        self.history_segment_num = modeling_config.history_segment_num
        self.history_segment_alpha = modeling_config.history_segment_alpha
        self.history_segment_decay = modeling_config.history_segment_decay
        self.event_prior_weight = modeling_config.event_prior_weight

        state_dim = config.robot_state_feature.shape[0]
        history_in_channels = state_dim * 3

        self.history_encoder = nn.Sequential(
            CausalConv1d(in_channels=history_in_channels, out_channels=64, kernel_size=3, dilation=1),
            nn.ReLU(),
            nn.GroupNorm(8, 64),
            CausalConv1d(in_channels=64, out_channels=128, kernel_size=3, dilation=2),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            CausalConv1d(in_channels=128, out_channels=256, kernel_size=3, dilation=4),
            nn.ReLU(),
        )

        self.segment_pool = EventAwareSegmentPooling(
            num_segments=self.history_segment_num,
            alpha=self.history_segment_alpha,
            feature_dim=256,
            event_prior_weight=self.event_prior_weight,
        )

        self.proj = nn.Sequential(
            nn.Linear(256 * 3, config.dim_model),
            nn.LayerNorm(config.dim_model),
        )

    def _build_history_features(self, x):
        """从绝对状态历史构造相对状态、速度、加速度和事件先验。

        Args:
            x: [B, T, state_dim]，按时间从旧到新排列，最后一帧是当前状态。

        Returns:
            history_features: [B, T, state_dim * 3]，rel_state/velocity/acceleration 拼接。
            event_prior: [B, T]，速度和加速度幅值构成的运动突变分数。
        """
        rel_state = x - x[:, -1:, :]

        velocity = torch.zeros_like(x)
        velocity[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]

        acceleration = torch.zeros_like(x)
        acceleration[:, 1:, :] = velocity[:, 1:, :] - velocity[:, :-1, :]

        event_prior = velocity.norm(dim=-1) + acceleration.norm(dim=-1)
        history_features = torch.cat([rel_state, velocity, acceleration], dim=-1)
        return history_features, event_prior

    def forward(self, x):
        """把历史状态序列编码为 ACT encoder 可用的事件感知 history tokens。

        Args:
            x: [B, T, state_dim]，来自 batch["history_obs_states"]。

        Returns:
            [B, num_segments, dim_model]，可直接作为多个 token 拼进 ACT encoder。
        """
        history_features, event_prior = self._build_history_features(x)
        history_features = history_features.permute(0, 2, 1)

        feat = self.history_encoder(history_features)
        pooled = self.segment_pool(feat, event_prior=event_prior)
        return self.proj(pooled)
