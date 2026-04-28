from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.customACT.history_obs_state.configuration_history_obs import HistoryConv1dConfig

if TYPE_CHECKING:
    from lerobot.policies.customACT.configuration_customACT import ACTConfig


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


class RecoveryHistoryTokenEncoder(nn.Module):
    """Failure-aware proprioceptive history encoder for ACT.

    这个模块只看数据集中已经存在的本体状态和动作，不需要人工阶段标签，也不需要
    力/触觉传感器。它把最近 H 步历史压缩成少量 recovery tokens，然后作为额外
    token 接到 ACT Transformer encoder 中。

    输入张量约定：
      - state_history: [B, H, state_dim] = [s_{t-H+1}, ..., s_t]
      - action_history: [B, H, action_dim] = [a_{t-H}, ..., a_{t-1}]
      - current_state: [B, state_dim] = s_t
      - history_mask: [B, H]，True 表示该位置是真实历史，不是 episode 开头 padding。

    输出：
      - recovery_tokens: [B, history_num_segments, dim_model]，追加到 ACT encoder。
      - aux_outputs: 事件分数、运动突变先验、历史动作预测、恢复倾向分数。
    """

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        dim_model: int,
        history_len: int = 32,
        history_num_segments: int = 4,
        history_hidden_dim: int = 256,
        history_conv_kernel_size: int = 3,
        history_conv_dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        history_dropout: float = 0.1,
        use_action_state_error: bool = True,
    ) -> None:
        super().__init__()
        assert history_len > 0, "history_len must be positive"
        assert history_num_segments > 0, "history_num_segments must be positive"
        assert history_len >= history_num_segments, "history_len must be >= history_num_segments"
        assert state_dim > 0 and action_dim > 0 and dim_model > 0

        event_hidden_dim = max(1, history_hidden_dim // 2)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dim_model = dim_model
        self.history_len = history_len
        self.history_num_segments = history_num_segments
        self.history_hidden_dim = history_hidden_dim
        self.use_action_state_error = use_action_state_error

        self.action_to_state = (
            nn.Identity() if action_dim == state_dim else nn.Linear(action_dim, state_dim)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim * 4, history_hidden_dim),
            nn.LayerNorm(history_hidden_dim),
            nn.GELU(),
        )
        self.conv_layers = nn.ModuleList(
            [
                CausalConv1d(
                    history_hidden_dim,
                    history_hidden_dim,
                    kernel_size=history_conv_kernel_size,
                    dilation=dilation,
                )
                for dilation in history_conv_dilations
            ]
        )
        self.conv_norms = nn.ModuleList(
            [nn.LayerNorm(history_hidden_dim) for _ in history_conv_dilations]
        )
        self.dropout = nn.Dropout(history_dropout)

        self.event_score_mlp = nn.Sequential(
            nn.LayerNorm(history_hidden_dim),
            nn.Linear(history_hidden_dim, event_hidden_dim),
            nn.GELU(),
            nn.Linear(event_hidden_dim, 1),
        )
        self.prior_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.segment_proj = nn.Sequential(
            nn.Linear(history_hidden_dim * 3, dim_model),
            nn.LayerNorm(dim_model),
            nn.GELU(),
            nn.Linear(dim_model, dim_model),
            nn.LayerNorm(dim_model),
        )
        self.hist_action_head = nn.Linear(dim_model, action_dim)
        self.recovery_score_head = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, 1),
        )

    def _build_history_features(
        self,
        state_history: torch.Tensor,
        action_history: torch.Tensor,
        current_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build per-timestep recovery features and a no-label event prior.

        对每一个历史时刻 i 构造四类信息：
          1. rel_i = s_i - s_t：历史状态相对当前状态的位移；
          2. vel_i = s_i - s_{i-1}：一阶速度，最左侧用 0 padding；
          3. acc_i = vel_i - vel_{i-1}：二阶加速度，最左侧用 0 padding；
          4. exec_error_i = a_{i-1} - s_i：动作和实际状态之间的执行偏差。

        如果 action_dim != state_dim，动作会先经过 Linear(action_dim, state_dim)
        投影到状态空间再相减。event_prior 不参与人工监督，只作为运动突变提示。

        Args:
            state_history: [B, H, state_dim].
            action_history: [B, H, action_dim].
            current_state: [B, state_dim].

        Returns:
            hist_feat: [B, H, 4 * state_dim].
            event_prior: [B, H].
        """
        rel = state_history - current_state.unsqueeze(1)

        vel = torch.zeros_like(state_history)
        vel[:, 1:] = state_history[:, 1:] - state_history[:, :-1]

        acc = torch.zeros_like(state_history)
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]

        if self.use_action_state_error:
            projected_action = self.action_to_state(action_history)
            exec_error = projected_action - state_history
        else:
            exec_error = torch.zeros_like(state_history)

        hist_feat = torch.cat([rel, vel, acc, exec_error], dim=-1)
        event_prior = vel.norm(dim=-1) + acc.norm(dim=-1) + exec_error.norm(dim=-1)
        return hist_feat, event_prior

    def _segment_boundaries(self, H: int) -> list[tuple[int, int]]:
        """Split H history steps into segment intervals [start, end).

        前 history_num_segments - 1 段长度为 H // history_num_segments，最后一段自动
        吃掉余数，因此 history_len 不能整除时也不会丢掉任何历史帧。
        """
        assert H >= self.history_num_segments
        base = H // self.history_num_segments
        boundaries = []
        start = 0
        for seg_idx in range(self.history_num_segments):
            end = H if seg_idx == self.history_num_segments - 1 else start + base
            boundaries.append((start, end))
            start = end
        return boundaries

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked temporal mean.

        Args:
            x: [B, L, D] segment features.
            mask: [B, L] validity mask.

        Returns:
            [B, D] mean feature. If a segment is fully padded, the denominator is clamped
            and the result becomes zero instead of NaN.
        """
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (x * mask_f).sum(dim=1) / denom

    def _masked_tail(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the last valid timestep feature in a segment.

        This captures the segment's endpoint state. If all positions in the segment are
        padding, the returned tail feature is [B, D] zeros.
        """
        B, _, D = x.shape
        lengths = mask.long().sum(dim=1)
        gather_idx = (lengths - 1).clamp_min(0).view(B, 1, 1).expand(B, 1, D)
        tail = x.gather(dim=1, index=gather_idx).squeeze(1)
        return torch.where(lengths.unsqueeze(-1) > 0, tail, torch.zeros_like(tail))

    def forward(
        self,
        state_history: torch.Tensor,
        action_history: torch.Tensor,
        current_state: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Encode recovery history into segment-level ACT tokens.

        Forward 流程：
          1. 构造 rel/vel/acc/exec_error 历史特征，得到 [B, H, 4 * state_dim]；
          2. Linear + LayerNorm + GELU 投影到 history_hidden_dim；
          3. 多层膨胀因果卷积编码时间上下文，只做左 padding，不看未来；
          4. learned_event_score + prior_scale * event_prior 得到事件 logits；
          5. 每段提取 trend/tail/event 三类特征并投影成 recovery token；
          6. 同时输出辅助头，用于训练日志和轻量辅助监督。

        Args:
            state_history: [B, H, state_dim].
            action_history: [B, H, action_dim].
            current_state: [B, state_dim].
            history_mask: [B, H], bool/float mask where 1 means real history.

        Returns:
            recovery_tokens: [B, history_num_segments, dim_model].
            aux_outputs: event scores, priors, first-action prediction, recovery score.
        """
        assert state_history.ndim == 3, f"state_history must be [B,H,state_dim], got {state_history.shape}"
        assert action_history.ndim == 3, f"action_history must be [B,H,action_dim], got {action_history.shape}"
        assert current_state.ndim == 2, f"current_state must be [B,state_dim], got {current_state.shape}"
        assert history_mask.ndim == 2, f"history_mask must be [B,H], got {history_mask.shape}"

        B, H, state_dim = state_history.shape
        assert H == self.history_len, f"Expected history_len={self.history_len}, got {H}"
        assert state_dim == self.state_dim, f"Expected state_dim={self.state_dim}, got {state_dim}"
        assert action_history.shape[:2] == (B, H)
        assert action_history.shape[-1] == self.action_dim
        assert current_state.shape == (B, self.state_dim)
        assert history_mask.shape == (B, H)

        mask = history_mask.to(device=state_history.device, dtype=torch.bool)
        hist_feat, event_prior = self._build_history_features(
            state_history, action_history, current_state
        )
        hist_feat = hist_feat * mask.unsqueeze(-1).to(dtype=hist_feat.dtype)

        x = self.input_proj(hist_feat)  # [B, H, history_hidden_dim]
        for conv, norm in zip(self.conv_layers, self.conv_norms, strict=True):
            # CausalConv1d receives [B, C, H] and returns [B, C, H]. Because the
            # convolution pads only on the left, x[:, t] can only depend on <= t.
            y = conv(x.transpose(1, 2)).transpose(1, 2)
            y = self.dropout(F.gelu(y))
            x = norm(x + y)
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        learned_event_score = self.event_score_mlp(x).squeeze(-1)  # [B, H]
        event_scores = learned_event_score + self.prior_scale * event_prior
        event_scores = event_scores.masked_fill(~mask, -1e4)

        segment_tokens = []
        for start, end in self._segment_boundaries(H):
            seg_x = x[:, start:end]
            seg_mask = mask[:, start:end]
            seg_scores = event_scores[:, start:end]

            # trend_feat: what generally happened in this interval.
            trend_feat = self._masked_mean(seg_x, seg_mask)
            # tail_feat: the latest valid feature in this interval.
            tail_feat = self._masked_tail(seg_x, seg_mask)

            # event_feat: softly attend to the most recovery-relevant moment in the interval.
            seg_weight = torch.softmax(seg_scores, dim=-1) * seg_mask.to(dtype=seg_x.dtype)
            seg_weight = seg_weight / seg_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            event_feat = torch.sum(seg_x * seg_weight.unsqueeze(-1), dim=1)

            segment_tokens.append(torch.cat([trend_feat, tail_feat, event_feat], dim=-1))

        recovery_tokens = self.segment_proj(torch.stack(segment_tokens, dim=1))
        token_summary = recovery_tokens.mean(dim=1)
        aux_outputs = {
            "event_scores": event_scores,
            "event_prior": event_prior,
            "hist_action_pred": self.hist_action_head(token_summary),
            "recovery_score": torch.sigmoid(self.recovery_score_head(token_summary)),
        }
        return recovery_tokens, aux_outputs


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
