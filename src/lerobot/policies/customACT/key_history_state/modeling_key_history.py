from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.customACT.history_obs_state.embedding_conv1d_history_obs import CausalConv1d


class KeyHistoryTokenEncoder(nn.Module):
    """Event-guided key historical state encoder for ACT.

    Inputs:
      - state_history: [B, H, state_dim] = [s_{t-H+1}, ..., s_t]
      - current_state: [B, state_dim] = s_t
      - history_mask: [B, H], True means real history rather than episode-start padding

    The encoder uses only state evolution features:
      - relative state: s_i - s_t
      - velocity: s_i - s_{i-1}
      - acceleration: velocity_i - velocity_{i-1}

    It intentionally does not use action-state execution error or recovery scores.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        dim_model: int,
        history_len: int = 64,
        num_segments: int = 4,
        hidden_dim: int = 256,
        conv_kernel_size: int = 3,
        conv_dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        prior_scale_init: float = 0.5,
        selection_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if history_len <= 0:
            raise ValueError("history_len must be positive")
        if num_segments <= 0:
            raise ValueError("num_segments must be positive")
        if history_len < num_segments:
            raise ValueError("history_len must be >= num_segments")
        if hidden_dim <= 0 or state_dim <= 0 or action_dim <= 0 or dim_model <= 0:
            raise ValueError("state_dim, action_dim, dim_model, and hidden_dim must be positive")
        if selection_temperature <= 0:
            raise ValueError("selection_temperature must be positive")

        event_hidden_dim = max(1, hidden_dim // 2)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dim_model = dim_model
        self.history_len = history_len
        self.num_segments = num_segments
        self.hidden_dim = hidden_dim
        self.selection_temperature = selection_temperature

        self.input_proj = nn.Sequential(
            nn.Linear(state_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.conv_layers = nn.ModuleList(
            [
                CausalConv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=conv_kernel_size,
                    dilation=dilation,
                )
                for dilation in conv_dilations
            ]
        )
        self.conv_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in conv_dilations])
        self.dropout = nn.Dropout(dropout)

        self.event_score_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, event_hidden_dim),
            nn.GELU(),
            nn.Linear(event_hidden_dim, 1),
        )
        self.prior_scale = nn.Parameter(torch.tensor(prior_scale_init, dtype=torch.float32))

        self.segment_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, dim_model),
            nn.LayerNorm(dim_model),
            nn.GELU(),
            nn.Linear(dim_model, dim_model),
            nn.LayerNorm(dim_model),
        )
        self.hist_action_head = nn.Linear(dim_model, action_dim)

    def _build_history_features(
        self,
        state_history: torch.Tensor,
        current_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rel = state_history - current_state.unsqueeze(1)

        vel = torch.zeros_like(state_history)
        vel[:, 1:] = state_history[:, 1:] - state_history[:, :-1]

        acc = torch.zeros_like(state_history)
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]

        hist_feat = torch.cat([rel, vel, acc], dim=-1)
        event_prior = vel.norm(dim=-1) + acc.norm(dim=-1)
        return hist_feat, event_prior

    def _segment_boundaries(self, H: int) -> list[tuple[int, int]]:
        base = H // self.num_segments
        boundaries = []
        start = 0
        for seg_idx in range(self.num_segments):
            end = H if seg_idx == self.num_segments - 1 else start + base
            boundaries.append((start, end))
            start = end
        return boundaries

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (x * mask_f).sum(dim=1) / denom

    def _masked_tail(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, _, D = x.shape
        lengths = mask.long().sum(dim=1)
        gather_idx = (lengths - 1).clamp_min(0).view(B, 1, 1).expand(B, 1, D)
        tail = x.gather(dim=1, index=gather_idx).squeeze(1)
        return torch.where(lengths.unsqueeze(-1) > 0, tail, torch.zeros_like(tail))

    def forward(
        self,
        *,
        state_history: torch.Tensor,
        current_state: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if state_history.ndim != 3:
            raise ValueError(f"state_history must be [B,H,state_dim], got {state_history.shape}")
        if current_state.ndim != 2:
            raise ValueError(f"current_state must be [B,state_dim], got {current_state.shape}")
        if history_mask.ndim != 2:
            raise ValueError(f"history_mask must be [B,H], got {history_mask.shape}")

        B, H, state_dim = state_history.shape
        if H != self.history_len:
            raise ValueError(f"Expected history_len={self.history_len}, got {H}")
        if state_dim != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {state_dim}")
        if current_state.shape != (B, self.state_dim):
            raise ValueError(f"Expected current_state shape {(B, self.state_dim)}, got {current_state.shape}")
        if history_mask.shape != (B, H):
            raise ValueError(f"Expected history_mask shape {(B, H)}, got {history_mask.shape}")

        mask = history_mask.to(device=state_history.device, dtype=torch.bool)
        hist_feat, event_prior = self._build_history_features(state_history, current_state)
        hist_feat = hist_feat * mask.unsqueeze(-1).to(dtype=hist_feat.dtype)

        x = self.input_proj(hist_feat)
        for conv, norm in zip(self.conv_layers, self.conv_norms, strict=True):
            y = conv(x.transpose(1, 2)).transpose(1, 2)
            y = self.dropout(F.gelu(y))
            x = norm(x + y)
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        learned_event_score = self.event_score_mlp(x).squeeze(-1)
        event_scores = learned_event_score + self.prior_scale * event_prior
        event_scores = event_scores.masked_fill(~mask, -1e4)

        segment_tokens = []
        key_weights = torch.zeros_like(event_scores)
        selected_indices = []
        for start, end in self._segment_boundaries(H):
            seg_x = x[:, start:end]
            seg_mask = mask[:, start:end]
            seg_scores = event_scores[:, start:end]

            trend_feat = self._masked_mean(seg_x, seg_mask)
            tail_feat = self._masked_tail(seg_x, seg_mask)

            seg_weight = torch.softmax(seg_scores / self.selection_temperature, dim=-1)
            seg_weight = seg_weight * seg_mask.to(dtype=seg_x.dtype)
            seg_weight = seg_weight / seg_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            key_feat = torch.sum(seg_x * seg_weight.unsqueeze(-1), dim=1)
            key_weights[:, start:end] = seg_weight

            selected = torch.argmax(seg_scores, dim=-1) + start
            selected = torch.where(seg_mask.any(dim=-1), selected, torch.zeros_like(selected))
            selected_indices.append(selected)

            segment_tokens.append(torch.cat([trend_feat, tail_feat, key_feat], dim=-1))

        key_history_tokens = self.segment_proj(torch.stack(segment_tokens, dim=1))
        token_summary = key_history_tokens.mean(dim=1)
        aux_outputs = {
            "event_scores": event_scores,
            "event_prior": event_prior,
            "key_weights": key_weights,
            "selected_indices": torch.stack(selected_indices, dim=1),
            "hist_action_pred": self.hist_action_head(token_summary),
        }
        return key_history_tokens, aux_outputs

