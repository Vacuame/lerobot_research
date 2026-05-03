from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


class CausalConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv1d(x)


def _load_key_history_encoder_class():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root
        / "src"
        / "lerobot"
        / "policies"
        / "customACT"
        / "key_history_state"
        / "modeling_key_history.py"
    )

    fake_history_module = types.ModuleType(
        "lerobot.policies.customACT.history_obs_state.embedding_conv1d_history_obs"
    )
    fake_history_module.CausalConv1d = CausalConv1d
    sys.modules[fake_history_module.__name__] = fake_history_module

    import importlib.util

    spec = importlib.util.spec_from_file_location("key_history_modeling", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.KeyHistoryTokenEncoder


def test_key_history_token_shapes():
    KeyHistoryTokenEncoder = _load_key_history_encoder_class()

    B, H, state_dim, action_dim, dim_model = 2, 32, 7, 7, 512
    encoder = KeyHistoryTokenEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        dim_model=dim_model,
        history_len=H,
        num_segments=4,
        hidden_dim=128,
        conv_kernel_size=3,
        conv_dilations=[1, 2, 4],
        dropout=0.1,
        prior_scale_init=0.5,
        selection_temperature=1.0,
    )

    state_history = torch.randn(B, H, state_dim)
    current_state = state_history[:, -1]
    history_mask = torch.ones(B, H, dtype=torch.bool)
    history_mask[0, :5] = False

    key_tokens, aux_outputs = encoder(
        state_history=state_history,
        current_state=current_state,
        history_mask=history_mask,
    )

    assert key_tokens.shape == (B, 4, dim_model)
    assert aux_outputs["event_scores"].shape == (B, H)
    assert aux_outputs["event_prior"].shape == (B, H)
    assert aux_outputs["key_weights"].shape == (B, H)
    assert aux_outputs["selected_indices"].shape == (B, 4)
    assert aux_outputs["hist_action_pred"].shape == (B, action_dim)


if __name__ == "__main__":
    test_key_history_token_shapes()
    print("KeyHistoryTokenEncoder shape test passed.")
