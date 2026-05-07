from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_recovery_adaptive_classes():
    repo_root = Path(__file__).resolve().parents[2]
    module_dir = (
        repo_root
        / "src"
        / "lerobot"
        / "policies"
        / "customACT"
        / "recovery_adaptive_chunking"
    )

    for name in [
        "lerobot",
        "lerobot.policies",
        "lerobot.policies.customACT",
        "lerobot.policies.customACT.recovery_adaptive_chunking",
    ]:
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        sys.modules[name] = pkg

    adaptive_pkg = types.ModuleType("lerobot.policies.customACT.adaptive_action_chunking")
    adaptive_pkg.__path__ = []

    class AdaptiveActionChunkingConfig:
        state_history_len = 64
        min_chunk_size = 8
        max_chunk_size = 0
        stable_chunk_multiplier = 1.5
        unstable_chunk_multiplier = 0.5
        volatility_low = 0.03
        volatility_high = 0.12
        acceleration_high = 0.10
        action_uncertainty_low = 0.03
        action_uncertainty_high = 0.15
        debug_print_every = 1
        debug_print_num_actions = 3
        debug_print_action_dims = 6

    adaptive_pkg.AdaptiveActionChunkingConfig = AdaptiveActionChunkingConfig
    adaptive_pkg.AdaptiveActionChunkingController = object
    adaptive_pkg.AdaptiveActionChunkingDecision = object
    sys.modules[adaptive_pkg.__name__] = adaptive_pkg
    adaptive_config_module = types.ModuleType(
        "lerobot.policies.customACT.adaptive_action_chunking.configuration_adaptive_action_chunking"
    )
    adaptive_config_module.AdaptiveActionChunkingConfig = AdaptiveActionChunkingConfig
    sys.modules[adaptive_config_module.__name__] = adaptive_config_module

    config_name = (
        "lerobot.policies.customACT.recovery_adaptive_chunking."
        "configuration_recovery_adaptive_chunking"
    )
    config_spec = importlib.util.spec_from_file_location(
        config_name, module_dir / "configuration_recovery_adaptive_chunking.py"
    )
    config_module = importlib.util.module_from_spec(config_spec)
    assert config_spec is not None and config_spec.loader is not None
    sys.modules[config_name] = config_module
    config_spec.loader.exec_module(config_module)

    modeling_name = (
        "lerobot.policies.customACT.recovery_adaptive_chunking."
        "modeling_recovery_adaptive_chunking"
    )
    modeling_spec = importlib.util.spec_from_file_location(
        modeling_name, module_dir / "modeling_recovery_adaptive_chunking.py"
    )
    modeling_module = importlib.util.module_from_spec(modeling_spec)
    assert modeling_spec is not None and modeling_spec.loader is not None
    sys.modules[modeling_name] = modeling_module
    modeling_spec.loader.exec_module(modeling_module)
    return config_module.RecoveryAdaptiveChunkingConfig, modeling_module


def test_future_action_correction_increases_recovery_target():
    RecoveryAdaptiveChunkingConfig, modeling = _load_recovery_adaptive_classes()
    cfg = RecoveryAdaptiveChunkingConfig(
        recovery_score_target_center=0.5,
        recovery_score_target_temperature=0.2,
        recovery_score_recent_steps=4,
    )

    B, H, K, D = 2, 8, 4, 3
    state_history = torch.zeros(B, H, D)
    action_history = torch.zeros(B, H, D)
    stable_future = torch.zeros(B, K, D)
    corrected_future = stable_future.clone()
    corrected_future[:, 0] = 4.0
    history_mask = torch.ones(B, H, dtype=torch.bool)
    action_is_pad = torch.zeros(B, K, dtype=torch.bool)

    stable = modeling.compute_recovery_score_target(
        state_history=state_history,
        action_history=action_history,
        future_actions=stable_future,
        history_mask=history_mask,
        action_is_pad=action_is_pad,
        config=cfg,
    )
    corrected = modeling.compute_recovery_score_target(
        state_history=state_history,
        action_history=action_history,
        future_actions=corrected_future,
        history_mask=history_mask,
        action_is_pad=action_is_pad,
        config=cfg,
    )

    assert corrected["future_action_correction"].mean() > stable["future_action_correction"].mean()
    assert corrected["target"].mean() > stable["target"].mean()


def test_recovery_score_loss_is_finite():
    RecoveryAdaptiveChunkingConfig, modeling = _load_recovery_adaptive_classes()
    cfg = RecoveryAdaptiveChunkingConfig(
        recovery_score_target_center=0.5,
        recovery_score_target_temperature=0.2,
    )

    B, H, K, D = 2, 8, 4, 3
    recovery_score = torch.full((B, 1), 0.5)
    state_history = torch.randn(B, H, D)
    action_history = torch.randn(B, H, D)
    future_actions = torch.randn(B, K, D)
    history_mask = torch.ones(B, H, dtype=torch.bool)
    action_is_pad = torch.zeros(B, K, dtype=torch.bool)

    loss, target_info = modeling.compute_recovery_score_loss(
        recovery_score=recovery_score,
        state_history=state_history,
        action_history=action_history,
        future_actions=future_actions,
        history_mask=history_mask,
        action_is_pad=action_is_pad,
        config=cfg,
    )

    assert torch.isfinite(loss)
    assert target_info["target"].shape == (B,)


if __name__ == "__main__":
    test_future_action_correction_increases_recovery_target()
    test_recovery_score_loss_is_finite()
    print("Recovery adaptive chunking tests passed.")
