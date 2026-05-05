from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor

from .configuration_adaptive_action_chunking import AdaptiveActionChunkingConfig


@dataclass
class AdaptiveActionChunkingDecision:
    chunk_size: int
    base_chunk_size: int
    old_action_weight: float
    state_volatility: float
    state_acceleration: float
    action_uncertainty: float
    recovery_score: float | None
    regime: str


class AdaptiveActionChunkingController:
    """Inference controller for history-aware adaptive action chunking.

    It deliberately stays outside the ACT network. The controller consumes normalized
    online states and the fixed-length ACT action chunk, then returns a conservative
    execution length and an old-plan weight for temporal aggregation.
    """

    def __init__(
        self,
        config: AdaptiveActionChunkingConfig,
        *,
        policy_chunk_size: int,
        policy_n_action_steps: int,
    ) -> None:
        self.config = config
        runtime_max = min(policy_chunk_size, policy_n_action_steps)
        if config.max_chunk_size > 0:
            runtime_max = min(runtime_max, config.max_chunk_size)
        self.max_chunk_size = max(1, runtime_max)
        self.min_chunk_size = min(max(1, config.min_chunk_size), self.max_chunk_size)
        self.state_history = deque([], maxlen=max(2, config.state_history_len))
        self.previous_chunk_size: int | None = None
        self.latest_decision: AdaptiveActionChunkingDecision | None = None
        self.ensembled_actions: Tensor | None = None

    def reset(self) -> None:
        self.state_history.clear()
        self.previous_chunk_size = None
        self.latest_decision = None
        self.ensembled_actions = None

    def observe_state(self, state: Tensor | None) -> None:
        if state is None:
            return
        state = state.detach()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape[0] != 1:
            return
        self.state_history.append(state[0].float().cpu())

    def decide(self, actions: Tensor, *, recovery_score: float | None = None) -> AdaptiveActionChunkingDecision:
        max_chunk_size = min(self.max_chunk_size, actions.shape[1])
        min_chunk_size = min(self.min_chunk_size, max_chunk_size)

        base_chunk_size, action_uncertainty = self._base_chunk_size_from_actions(
            actions,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )
        state_volatility, state_acceleration = self._state_motion_stats()

        high_recovery = recovery_score is not None and recovery_score >= self.config.recovery_score_high
        high_action_uncertainty = action_uncertainty >= self.config.action_uncertainty_high
        unstable = (
            state_volatility >= self.config.volatility_high
            or state_acceleration >= self.config.acceleration_high
            or high_recovery
            or high_action_uncertainty
        )
        stable = (
            state_volatility <= self.config.volatility_low
            and state_acceleration < self.config.acceleration_high * 0.5
            and action_uncertainty <= self.config.action_uncertainty_low
            and not high_recovery
        )

        if unstable:
            regime = "unstable"
            scaled_chunk_size = round(base_chunk_size * self.config.unstable_chunk_multiplier)
            old_action_weight = self.config.unstable_old_action_weight
        elif stable:
            regime = "stable"
            scaled_chunk_size = round(base_chunk_size * self.config.stable_chunk_multiplier)
            old_action_weight = self.config.stable_old_action_weight
        else:
            regime = "nominal"
            scaled_chunk_size = base_chunk_size
            old_action_weight = self.config.nominal_old_action_weight

        chunk_size = self._smooth_chunk_size(
            self._clip_chunk_size(scaled_chunk_size, min_chunk_size, max_chunk_size),
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )
        old_action_weight = float(
            min(
                max(old_action_weight, self.config.min_old_action_weight),
                self.config.max_old_action_weight,
            )
        )

        decision = AdaptiveActionChunkingDecision(
            chunk_size=chunk_size,
            base_chunk_size=base_chunk_size,
            old_action_weight=old_action_weight,
            state_volatility=state_volatility,
            state_acceleration=state_acceleration,
            action_uncertainty=action_uncertainty,
            recovery_score=recovery_score,
            regime=regime,
        )
        self.latest_decision = decision
        return decision

    def update_temporal_ensemble(self, actions: Tensor, *, old_action_weight: float) -> Tensor:
        """Blend overlapping future actions with a dynamic old-plan weight."""
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
        else:
            overlap = min(self.ensembled_actions.shape[1], actions.shape[1])
            old_weight = torch.as_tensor(
                old_action_weight,
                dtype=actions.dtype,
                device=actions.device,
            )
            merged = (
                old_weight * self.ensembled_actions[:, :overlap]
                + (1.0 - old_weight) * actions[:, :overlap]
            )
            if actions.shape[1] > overlap:
                self.ensembled_actions = torch.cat([merged, actions[:, overlap:]], dim=1)
            else:
                self.ensembled_actions = merged

        action = self.ensembled_actions[:, 0]
        self.ensembled_actions = self.ensembled_actions[:, 1:]
        if self.ensembled_actions.shape[1] == 0:
            self.ensembled_actions = None
        return action

    def _state_motion_stats(self) -> tuple[float, float]:
        if len(self.state_history) < 2:
            return 0.0, 0.0
        states = torch.stack(list(self.state_history), dim=0)
        velocity = states[1:] - states[:-1]
        volatility = torch.sqrt(velocity.pow(2).mean(dim=-1).clamp_min(0.0)).mean()
        if velocity.shape[0] < 2:
            return float(volatility.item()), 0.0
        acceleration = velocity[1:] - velocity[:-1]
        acceleration_score = torch.sqrt(acceleration.pow(2).mean(dim=-1).clamp_min(0.0)).mean()
        return float(volatility.item()), float(acceleration_score.item())

    def _base_chunk_size_from_actions(
        self,
        actions: Tensor,
        *,
        min_chunk_size: int,
        max_chunk_size: int,
    ) -> tuple[int, float]:
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape [B, K, action_dim], got {actions.shape}")
        if actions.shape[1] <= min_chunk_size:
            return actions.shape[1], 0.0

        action_seq = actions[0, :max_chunk_size].detach().float()
        velocity = action_seq[1:] - action_seq[:-1]
        step_score = torch.sqrt(velocity.pow(2).mean(dim=-1).clamp_min(0.0))

        if action_seq.shape[0] >= 3 and self.config.action_curvature_weight > 0:
            curvature = action_seq[2:] - 2 * action_seq[1:-1] + action_seq[:-2]
            curvature_score = torch.sqrt(curvature.pow(2).mean(dim=-1).clamp_min(0.0))
            step_score = step_score.clone()
            step_score[1:] = step_score[1:] + self.config.action_curvature_weight * curvature_score

        uncertainty = float(step_score.mean().item()) if step_score.numel() else 0.0
        if step_score.numel() < 2 or torch.all(step_score < 1e-6):
            return max_chunk_size, uncertainty

        score_change = (step_score[1:] - step_score[:-1]).abs()
        candidate_positions = torch.arange(2, max_chunk_size, device=score_change.device)
        valid = (candidate_positions >= min_chunk_size) & (candidate_positions <= max_chunk_size)
        if not valid.any():
            return max_chunk_size, uncertainty

        valid_changes = score_change[valid]
        valid_positions = candidate_positions[valid]
        base_chunk_size = int(valid_positions[torch.argmax(valid_changes)].item())
        return self._clip_chunk_size(base_chunk_size, min_chunk_size, max_chunk_size), uncertainty

    def _smooth_chunk_size(
        self,
        chunk_size: int,
        *,
        min_chunk_size: int,
        max_chunk_size: int,
    ) -> int:
        if self.previous_chunk_size is None:
            self.previous_chunk_size = chunk_size
            return chunk_size

        smoothing = min(max(self.config.chunk_smoothing, 0.0), 1.0)
        smoothed = round((1.0 - smoothing) * self.previous_chunk_size + smoothing * chunk_size)
        if self.config.max_chunk_delta > 0:
            max_delta = self.config.max_chunk_delta
            smoothed = min(max(smoothed, self.previous_chunk_size - max_delta), self.previous_chunk_size + max_delta)
        smoothed = self._clip_chunk_size(smoothed, min_chunk_size, max_chunk_size)
        self.previous_chunk_size = smoothed
        return smoothed

    @staticmethod
    def _clip_chunk_size(chunk_size: int, min_chunk_size: int, max_chunk_size: int) -> int:
        return int(min(max(chunk_size, min_chunk_size), max_chunk_size))
