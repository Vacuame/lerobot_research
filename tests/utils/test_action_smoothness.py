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

from lerobot.utils.action_smoothness import ActionSmoothnessTracker, format_action_smoothness


def test_action_smoothness_stays_high_for_constant_velocity():
    tracker = ActionSmoothnessTracker(ema_alpha=1.0)

    tracker.update({"joint_0": 0.0, "joint_1": 0.0})
    tracker.update({"joint_0": 1.0, "joint_1": 1.0})
    metrics = tracker.update({"joint_0": 2.0, "joint_1": 2.0})

    assert metrics.ready
    assert metrics.delta_rms == 1.0
    assert metrics.accel_rms == 0.0
    assert metrics.score == 100.0


def test_action_smoothness_drops_for_abrupt_action_change():
    tracker = ActionSmoothnessTracker(ema_alpha=1.0)

    tracker.update({"joint_0": 0.0})
    tracker.update({"joint_0": 1.0})
    metrics = tracker.update({"joint_0": -1.0})

    assert metrics.ready
    assert metrics.delta_rms == 2.0
    assert metrics.accel_rms == 3.0
    assert metrics.score < 50.0


def test_action_smoothness_uses_observed_robot_motion_when_available():
    tracker = ActionSmoothnessTracker(ema_alpha=1.0)

    tracker.update({"joint_0.pos": 0.0}, observation={"joint_0.pos": 0.0})
    tracker.update({"joint_0.pos": 100.0}, observation={"joint_0.pos": 1.0})
    metrics = tracker.update({"joint_0.pos": -100.0}, observation={"joint_0.pos": 2.0})

    assert metrics.ready
    assert metrics.delta_rms == 1.0
    assert metrics.accel_rms == 0.0
    assert metrics.score == 100.0
    assert metrics.tracking_error_rms > 0.0


def test_action_smoothness_reports_each_joint_score_separately():
    tracker = ActionSmoothnessTracker(ema_alpha=1.0)

    tracker.update(
        {"joint_0.pos": 0.0, "joint_1.pos": 0.0},
        observation={"joint_0.pos": 0.0, "joint_1.pos": 0.0},
    )
    tracker.update(
        {"joint_0.pos": 1.0, "joint_1.pos": 1.0},
        observation={"joint_0.pos": 1.0, "joint_1.pos": 1.0},
    )
    metrics = tracker.update(
        {"joint_0.pos": 2.0, "joint_1.pos": -1.0},
        observation={"joint_0.pos": 2.0, "joint_1.pos": -1.0},
    )
    summary = tracker.summary()

    assert metrics.joint_scores["joint_0.pos"] == 100.0
    assert metrics.joint_scores["joint_1.pos"] == 40.0
    assert metrics.score == 70.0
    assert summary.avg_joint_scores["joint_0.pos"] == 100.0
    assert summary.avg_joint_scores["joint_1.pos"] == 40.0


def test_action_smoothness_summary_averages_ready_samples():
    tracker = ActionSmoothnessTracker(ema_alpha=1.0)

    tracker.update({"joint_0": 0.0})
    tracker.update({"joint_0": 1.0})
    tracker.update({"joint_0": 2.0})
    tracker.update({"joint_0": 1.0})
    summary = tracker.summary()

    assert summary.updates == 4
    assert summary.samples == 2
    assert summary.action_dim == 1
    assert summary.avg_delta_rms == 1.0
    assert summary.avg_accel_rms == 1.0
    assert summary.avg_score == 100 / 1.5


def test_action_smoothness_reports_untracked_targets_separately():
    tracker = ActionSmoothnessTracker(ema_alpha=1.0, tracking_error_scale=5.0)

    tracker.update({"joint_0.pos": 0.0}, observation={"joint_0.pos": 0.0})
    tracker.update({"joint_0.pos": 10.0}, observation={"joint_0.pos": 0.0})
    tracker.update({"joint_0.pos": 10.0}, observation={"joint_0.pos": 0.0})
    metrics = tracker.update({"joint_0.pos": 10.0}, observation={"joint_0.pos": 0.0})

    assert metrics.tracking_error_rms == 10.0
    assert metrics.score == 100.0


def test_action_smoothness_does_not_penalize_fast_tracked_motion():
    tracker = ActionSmoothnessTracker(ema_alpha=1.0, tracking_error_scale=5.0)

    tracker.update({"joint_0.pos": 0.0}, observation={"joint_0.pos": 0.0})
    tracker.update({"joint_0.pos": 10.0}, observation={"joint_0.pos": 5.0})
    tracker.update({"joint_0.pos": 20.0}, observation={"joint_0.pos": 15.0})
    metrics = tracker.update({"joint_0.pos": 30.0}, observation={"joint_0.pos": 25.0})

    assert metrics.delta_rms == 10.0
    assert metrics.accel_rms == 0.0
    assert metrics.tracking_error_rms == 0.0
    assert metrics.score == 100.0


def test_format_action_smoothness_contains_live_metrics():
    tracker = ActionSmoothnessTracker()
    metrics = tracker.update({"joint_0": 0.0})

    formatted = format_action_smoothness(metrics)

    assert "smooth:" in formatted
    assert "d_rms:" in formatted
    assert "a_rms:" in formatted
    assert "track:" in formatted
