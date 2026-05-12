#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from math import sqrt
from typing import Any


@dataclass(frozen=True)
class ActionSmoothness:
    """Per-step smoothness metrics computed from observed robot motion."""

    score: float
    delta_rms: float
    accel_rms: float
    tracking_error_rms: float
    action_dim: int
    ready: bool
    joint_scores: dict[str, float]


@dataclass(frozen=True)
class ActionSmoothnessSummary:
    """Average smoothness metrics for one control segment."""

    avg_score: float
    avg_delta_rms: float
    avg_accel_rms: float
    avg_tracking_error_rms: float
    avg_joint_scores: dict[str, float]
    samples: int
    updates: int
    action_dim: int


class ActionSmoothnessTracker:
    """Tracks robot motion smoothness using first and second finite differences.

    The score is a bounded heuristic for live comparison:
    - 100 means the observed robot velocity is steady.
    - Lower values mean the observed robot motion changed abruptly.

    Raw metrics are also exposed:
    - delta_rms: RMS(state_t - state_t-1)
    - accel_rms: RMS(delta_t - delta_t-1)
    - tracking_error_rms: residual target error after allowing for observed joint motion.

    Observation state is used for the smoothness score when available, so the metric reflects the path the
    robot actually moved through. If no usable observation state is provided, it falls back to the action
    command vector so callers without robot feedback still get a best-effort value.
    """

    def __init__(
        self,
        *,
        ema_alpha: float = 0.2,
        score_floor: float = 1.0,
        tracking_error_scale: float = 5.0,
        response_lag_allowance: float = 2.0,
    ):
        if not 0 < ema_alpha <= 1:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}.")
        if score_floor <= 0:
            raise ValueError(f"score_floor must be > 0, got {score_floor}.")
        if tracking_error_scale <= 0:
            raise ValueError(f"tracking_error_scale must be > 0, got {tracking_error_scale}.")
        if response_lag_allowance < 0:
            raise ValueError(f"response_lag_allowance must be >= 0, got {response_lag_allowance}.")

        self.ema_alpha = ema_alpha
        self.score_floor = score_floor
        self.tracking_error_scale = tracking_error_scale
        self.response_lag_allowance = response_lag_allowance
        self._previous_motion: list[float] | None = None
        self._motion_labels: list[str] = []
        self._previous_observation_dict: dict[str, Any] | None = None
        self._previous_delta: list[float] | None = None
        self._delta_rms_ema: float | None = None
        self._accel_rms_ema: float | None = None
        self._joint_delta_ema: list[float | None] = []
        self._joint_accel_ema: list[float | None] = []
        self._tracking_error_rms_ema: float | None = None
        self._score_sum = 0.0
        self._delta_rms_sum = 0.0
        self._accel_rms_sum = 0.0
        self._tracking_error_rms_sum = 0.0
        self._joint_score_sums: dict[str, float] = {}
        self._joint_samples: dict[str, int] = {}
        self._samples = 0
        self._updates = 0
        self._action_dim = 0

    def reset(self) -> None:
        self._previous_motion = None
        self._motion_labels = []
        self._previous_observation_dict = None
        self._previous_delta = None
        self._delta_rms_ema = None
        self._accel_rms_ema = None
        self._joint_delta_ema = []
        self._joint_accel_ema = []
        self._tracking_error_rms_ema = None
        self._score_sum = 0.0
        self._delta_rms_sum = 0.0
        self._accel_rms_sum = 0.0
        self._tracking_error_rms_sum = 0.0
        self._joint_score_sums = {}
        self._joint_samples = {}
        self._samples = 0
        self._updates = 0
        self._action_dim = 0

    def update(self, action: dict[str, Any], observation: dict[str, Any] | None = None) -> ActionSmoothness:
        action_labels, action_vector = _action_to_named_vector(action)
        motion_labels, motion_vector = (
            _observation_to_state_named_vector(observation) if observation is not None else ([], [])
        )
        if not motion_vector:
            motion_labels = action_labels
            motion_vector = action_vector

        if not motion_vector:
            return ActionSmoothness(
                score=100.0,
                delta_rms=0.0,
                accel_rms=0.0,
                tracking_error_rms=0.0,
                action_dim=0,
                ready=False,
                joint_scores={},
            )

        self._updates += 1
        self._action_dim = len(motion_vector)

        if self._previous_motion is None or motion_labels != self._motion_labels:
            self._motion_labels = motion_labels
            self._previous_motion = motion_vector
            self._previous_observation_dict = dict(observation) if observation is not None else None
            self._previous_delta = None
            self._delta_rms_ema = None
            self._accel_rms_ema = None
            self._joint_delta_ema = [None] * len(motion_vector)
            self._joint_accel_ema = [None] * len(motion_vector)
            return ActionSmoothness(
                score=100.0,
                delta_rms=0.0,
                accel_rms=0.0,
                tracking_error_rms=0.0,
                action_dim=len(motion_vector),
                ready=False,
                joint_scores=dict.fromkeys(motion_labels, 100.0),
            )

        delta = [cur - prev for cur, prev in zip(motion_vector, self._previous_motion, strict=True)]
        delta_rms = _rms(delta)

        if self._previous_delta is None:
            accel_rms = 0.0
            accel = [0.0] * len(delta)
            ready = False
        else:
            accel = [cur - prev for cur, prev in zip(delta, self._previous_delta, strict=True)]
            accel_rms = _rms(accel)
            ready = True

        self._delta_rms_ema = _ema(self._delta_rms_ema, delta_rms, self.ema_alpha)
        self._accel_rms_ema = _ema(self._accel_rms_ema, accel_rms, self.ema_alpha)

        joint_scores: dict[str, float] = {}
        for i, label in enumerate(motion_labels):
            self._joint_delta_ema[i] = _ema(self._joint_delta_ema[i], abs(delta[i]), self.ema_alpha)
            self._joint_accel_ema[i] = _ema(self._joint_accel_ema[i], abs(accel[i]), self.ema_alpha)
            score_scale = max(self.score_floor, self._joint_delta_ema[i])
            joint_scores[label] = 100.0 / (1.0 + self._joint_accel_ema[i] / score_scale)
        motion_score = _mean(list(joint_scores.values()))

        tracking_error = _response_error_vector(
            action,
            observation,
            self._previous_observation_dict,
            lag_allowance=self.response_lag_allowance,
        )
        tracking_error_rms = _rms(tracking_error)
        if tracking_error:
            self._tracking_error_rms_ema = _ema(
                self._tracking_error_rms_ema, tracking_error_rms, self.ema_alpha
            )
        else:
            self._tracking_error_rms_ema = _ema(self._tracking_error_rms_ema, 0.0, self.ema_alpha)

        metrics = ActionSmoothness(
            score=motion_score,
            delta_rms=self._delta_rms_ema,
            accel_rms=self._accel_rms_ema,
            tracking_error_rms=self._tracking_error_rms_ema,
            action_dim=len(motion_vector),
            ready=ready,
            joint_scores=joint_scores,
        )
        self._previous_motion = motion_vector
        self._previous_observation_dict = dict(observation) if observation is not None else None
        self._previous_delta = delta
        if metrics.ready:
            self._score_sum += metrics.score
            self._delta_rms_sum += metrics.delta_rms
            self._accel_rms_sum += metrics.accel_rms
            self._tracking_error_rms_sum += metrics.tracking_error_rms
            for label, score in metrics.joint_scores.items():
                self._joint_score_sums[label] = self._joint_score_sums.get(label, 0.0) + score
                self._joint_samples[label] = self._joint_samples.get(label, 0) + 1
            self._samples += 1

        return metrics

    def summary(self) -> ActionSmoothnessSummary:
        if self._samples == 0:
            return ActionSmoothnessSummary(
                avg_score=100.0,
                avg_delta_rms=0.0,
                avg_accel_rms=0.0,
                avg_tracking_error_rms=0.0,
                avg_joint_scores=dict.fromkeys(self._motion_labels, 100.0),
                samples=0,
                updates=self._updates,
                action_dim=self._action_dim,
            )

        return ActionSmoothnessSummary(
            avg_score=self._score_sum / self._samples,
            avg_delta_rms=self._delta_rms_sum / self._samples,
            avg_accel_rms=self._accel_rms_sum / self._samples,
            avg_tracking_error_rms=self._tracking_error_rms_sum / self._samples,
            avg_joint_scores={
                label: self._joint_score_sums[label] / self._joint_samples[label]
                for label in self._motion_labels
                if self._joint_samples.get(label, 0) > 0
            },
            samples=self._samples,
            updates=self._updates,
            action_dim=self._action_dim,
        )


def format_action_smoothness(metrics: ActionSmoothness) -> str:
    return (
        f"smooth: {metrics.score:5.1f}/100 | "
        f"d_rms: {metrics.delta_rms:.3f} | "
        f"a_rms: {metrics.accel_rms:.3f} | "
        f"track: {metrics.tracking_error_rms:.3f}"
    )


def _action_to_named_vector(action: dict[str, Any]) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    values: list[float] = []
    for key in sorted(action):
        key_values = _value_to_floats(action[key])
        labels.extend(_value_labels(key, len(key_values)))
        values.extend(key_values)
    return labels, values


def _observation_to_state_vector(observation: dict[str, Any] | None) -> list[float]:
    _, values = _observation_to_state_named_vector(observation)
    return values


def _observation_to_state_named_vector(observation: dict[str, Any] | None) -> tuple[list[str], list[float]]:
    if observation is None:
        return [], []

    labels: list[str] = []
    values: list[float] = []
    for key, value in observation.items():
        if key.endswith(".pos"):
            key_values = _value_to_floats(value)
            labels.extend(_value_labels(key, len(key_values)))
            values.extend(key_values)
    if values:
        return labels, values

    for key in ("observation.state", "state"):
        if key in observation:
            values = _value_to_floats(observation[key])
            return _value_labels(key, len(values)), values

    return [], []


def _value_labels(base_label: str, count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return [base_label]
    return [f"{base_label}_{i}" for i in range(count)]


def _response_error_vector(
    target: dict[str, Any],
    measurement: dict[str, Any] | None,
    previous_measurement: dict[str, Any] | None,
    *,
    lag_allowance: float,
) -> list[float]:
    if measurement is None or previous_measurement is None:
        return []

    errors: list[float] = []
    for key in sorted(target):
        if key not in measurement or key not in previous_measurement:
            continue
        target_values = _value_to_floats(target[key])
        measured_values = _value_to_floats(measurement[key])
        previous_measured_values = _value_to_floats(previous_measurement[key])
        if len(target_values) != len(measured_values) or len(measured_values) != len(previous_measured_values):
            continue
        for target_value, measured_value, previous_measured_value in zip(
            target_values, measured_values, previous_measured_values, strict=True
        ):
            target_error = abs(target_value - measured_value)
            observed_motion = abs(measured_value - previous_measured_value)
            errors.append(max(0.0, target_error - lag_allowance * observed_motion))
    return errors


def _value_to_floats(value: Any) -> list[float]:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "reshape") and not isinstance(value, str):
            return [float(v) for v in value.reshape(-1)]
        if hasattr(value, "item"):
            return [float(value.item())]
        if isinstance(value, (list, tuple)):
            floats: list[float] = []
            for item in value:
                floats.extend(_value_to_floats(item))
            return floats
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _ema(previous: float | None, current: float, alpha: float) -> float:
    if previous is None:
        return current
    return alpha * current + (1 - alpha) * previous


def _mean(values: list[float]) -> float:
    if not values:
        return 100.0
    return sum(values) / len(values)


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return sqrt(sum(value * value for value in values) / len(values))
