from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.customACT.adaptive_action_chunking import (
    AdaptiveActionChunkingController,
    AdaptiveActionChunkingDecision,
)
from lerobot.policies.customACT.recovery_adaptive_chunking.configuration_recovery_adaptive_chunking import (
    RecoveryAdaptiveChunkingConfig,
)


def _rms(x: Tensor) -> Tensor:
    return torch.sqrt(x.pow(2).mean(dim=-1).clamp_min(0.0))


def _masked_recent_mean(values: Tensor, mask: Tensor, recent_steps: int) -> Tensor:
    if recent_steps > 0:
        values = values[:, -recent_steps:]
        mask = mask[:, -recent_steps:]
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (values * mask_f).sum(dim=1) / denom


def _last_valid(history: Tensor, mask: Tensor) -> Tensor:
    batch_size, _, dim = history.shape
    lengths = mask.long().sum(dim=1)
    gather_idx = (lengths - 1).clamp_min(0).view(batch_size, 1, 1).expand(batch_size, 1, dim)
    values = history.gather(dim=1, index=gather_idx).squeeze(1)
    return torch.where(lengths.unsqueeze(-1) > 0, values, torch.zeros_like(values))


def compute_recovery_score_target(
    *,
    state_history: Tensor,
    action_history: Tensor,
    future_actions: Tensor,
    history_mask: Tensor,
    config: RecoveryAdaptiveChunkingConfig,
    action_is_pad: Tensor | None = None,
) -> dict[str, Tensor]:
    """Create a no-label recovery score target from ACT training data.

    The target is high when recent history shows command/state mismatch,
    unstable state motion, or a strong expert correction in the next action.
    It uses only fields already present in an ACT training batch.
    """
    if state_history.ndim != 3:
        raise ValueError(f"state_history must be [B,H,state_dim], got {state_history.shape}")
    if action_history.ndim != 3:
        raise ValueError(f"action_history must be [B,H,action_dim], got {action_history.shape}")
    if future_actions.ndim != 3:
        raise ValueError(f"future_actions must be [B,K,action_dim], got {future_actions.shape}")
    if history_mask.ndim != 2:
        raise ValueError(f"history_mask must be [B,H], got {history_mask.shape}")

    mask = history_mask.to(device=state_history.device, dtype=torch.bool)

    velocity = torch.zeros_like(state_history)
    velocity[:, 1:] = state_history[:, 1:] - state_history[:, :-1]
    acceleration = torch.zeros_like(state_history)
    acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
    state_motion = _rms(velocity) + _rms(acceleration)

    if config.use_action_state_error and action_history.shape[-1] == state_history.shape[-1]:
        exec_error = _rms(action_history - state_history)
    else:
        exec_error = torch.zeros_like(state_motion)

    recent_steps = config.recovery_score_recent_steps
    recent_exec_error = _masked_recent_mean(exec_error, mask, recent_steps)
    recent_state_motion = _masked_recent_mean(state_motion, mask, recent_steps)

    last_action = _last_valid(action_history, mask)
    first_future_action = future_actions[:, 0]
    future_action_correction = _rms(first_future_action - last_action)

    if future_actions.shape[1] > 1 and config.recovery_future_action_curvature_weight > 0:
        second_future_action = future_actions[:, 1]
        future_curvature = _rms(second_future_action - 2.0 * first_future_action + last_action)
        if action_is_pad is not None and action_is_pad.shape[1] > 1:
            second_valid = (~action_is_pad[:, 1]).to(device=future_actions.device, dtype=future_actions.dtype)
            future_curvature = future_curvature * second_valid
        future_action_correction = (
            future_action_correction
            + config.recovery_future_action_curvature_weight * future_curvature
        )

    if action_is_pad is not None:
        first_valid = (~action_is_pad[:, 0]).to(device=future_actions.device, dtype=future_actions.dtype)
        future_action_correction = future_action_correction * first_valid

    raw_score = (
        config.recovery_exec_error_weight * recent_exec_error
        + config.recovery_state_motion_weight * recent_state_motion
        + config.recovery_future_action_correction_weight * future_action_correction
    )
    target = torch.sigmoid(
        (raw_score - config.recovery_score_target_center)
        / config.recovery_score_target_temperature
    )

    return {
        "target": target.detach(),
        "raw_score": raw_score.detach(),
        "recent_exec_error": recent_exec_error.detach(),
        "recent_state_motion": recent_state_motion.detach(),
        "future_action_correction": future_action_correction.detach(),
    }


def compute_recovery_score_loss(
    *,
    recovery_score: Tensor,
    state_history: Tensor,
    action_history: Tensor,
    future_actions: Tensor,
    history_mask: Tensor,
    config: RecoveryAdaptiveChunkingConfig,
    action_is_pad: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    target_info = compute_recovery_score_target(
        state_history=state_history,
        action_history=action_history,
        future_actions=future_actions,
        history_mask=history_mask,
        config=config,
        action_is_pad=action_is_pad,
    )
    pred = recovery_score.squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
    target = target_info["target"].to(device=pred.device, dtype=pred.dtype)
    bce = F.binary_cross_entropy(pred, target, reduction="none")

    if action_is_pad is not None:
        valid = (~action_is_pad[:, 0]).to(device=pred.device, dtype=pred.dtype)
        loss = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    else:
        loss = bce.mean()

    return loss, target_info


__all__ = [
    "AdaptiveActionChunkingController",
    "AdaptiveActionChunkingDecision",
    "compute_recovery_score_loss",
    "compute_recovery_score_target",
]
