import pytest
import torch

from lerobot.policies.customACT.mask_weight.configuration_mask_weight import MaskWeightConfig
from lerobot.policies.customACT.mask_weight.mask_weight import MaskGuidedVisualAdapter


def test_mask_guided_visual_adapter_shapes_and_gradient():
    torch.manual_seed(0)
    config = MaskWeightConfig(adapter_hidden_dim=4, mask_dropout_p=0.0, use_target_tokens=False)
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)

    features = torch.randn(2, 8, 5, 6, requires_grad=True)
    mask = torch.zeros(2, 1, 5, 6)
    mask[:, :, 1:4, 2:5] = 0.8

    guided_features, target_tokens, target_pos_embed = adapter(features, mask)

    assert guided_features.shape == features.shape
    assert target_tokens is None
    assert target_pos_embed is None

    guided_features.mean().backward()
    assert adapter.gate_scale.grad is not None


def test_mask_guided_visual_adapter_target_tokens():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=0.0,
        use_target_tokens=True,
        num_target_tokens=3,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)

    features = torch.randn(2, 8, 5, 6)
    mask = torch.zeros(2, 1, 5, 6)
    mask[:, :, 1:4, 2:5] = 1.0

    guided_features, target_tokens, target_pos_embed = adapter(features, mask)

    assert guided_features.shape == features.shape
    assert target_tokens.shape == (3, 2, 8)
    assert target_pos_embed.shape == (3, 1, 8)


def test_mask_weight_config_rejects_invalid_mode():
    with pytest.raises(ValueError):
        MaskWeightConfig(mode="bad_mode")
