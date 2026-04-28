from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_recovery_encoder_class():
    """Load RecoveryHistoryTokenEncoder without importing the full lerobot package tree."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root
        / "src"
        / "lerobot"
        / "policies"
        / "customACT"
        / "history_obs_state"
        / "embedding_conv1d_history_obs.py"
    )

    fake_config_module = types.ModuleType(
        "lerobot.policies.customACT.history_obs_state.configuration_history_obs"
    )

    class HistoryConv1dConfig:  # noqa: D401 - only needed to satisfy the module import.
        """Minimal stub for direct file loading in this shape test."""

    fake_config_module.HistoryConv1dConfig = HistoryConv1dConfig
    sys.modules[fake_config_module.__name__] = fake_config_module

    spec = importlib.util.spec_from_file_location("recovery_history_embedding", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.RecoveryHistoryTokenEncoder


def test_recovery_history_token_shapes():
    RecoveryHistoryTokenEncoder = _load_recovery_encoder_class()

    B, H, state_dim, action_dim, dim_model = 2, 32, 7, 7, 512
    encoder = RecoveryHistoryTokenEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        dim_model=dim_model,
        history_len=H,
        history_num_segments=4,
        history_hidden_dim=256,
        history_conv_kernel_size=3,
        history_conv_dilations=[1, 2, 4, 8],
        history_dropout=0.1,
        use_action_state_error=True,
    )

    state_history = torch.randn(B, H, state_dim)
    action_history = torch.randn(B, H, action_dim)
    current_state = state_history[:, -1]
    history_mask = torch.ones(B, H, dtype=torch.bool)
    history_mask[0, :3] = False

    recovery_tokens, aux_outputs = encoder(
        state_history=state_history,
        action_history=action_history,
        current_state=current_state,
        history_mask=history_mask,
    )

    assert recovery_tokens.shape == (B, 4, dim_model)
    assert aux_outputs["event_scores"].shape == (B, H)
    assert aux_outputs["event_prior"].shape == (B, H)
    assert aux_outputs["hist_action_pred"].shape == (B, action_dim)
    assert aux_outputs["recovery_score"].shape == (B, 1)


if __name__ == "__main__":
    test_recovery_history_token_shapes()
    print("RecoveryHistoryTokenEncoder shape test passed.")
