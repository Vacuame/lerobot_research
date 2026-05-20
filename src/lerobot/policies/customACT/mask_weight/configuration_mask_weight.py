from dataclasses import dataclass


@dataclass
class MaskWeightConfig:
    # "adapter" injects YOLO masks as visual-token guidance.
    # "legacy_multiply" keeps the old fixed feature multiplication for ablations.
    mode: str = "adapter"

    # Legacy multiply: features = features * (beta + alpha * mask).
    alpha: float = 1.0
    beta: float = 0.5

    # Mask-guided adapter switches.
    use_spatial_embedding: bool = True
    use_residual_gate: bool = True
    use_target_tokens: bool = False

    # Mask-guided adapter hyperparameters.
    adapter_hidden_dim: int = 128
    gate_init: float = 0.1
    context_dilation: int = 3
    mask_dropout_p: float = 0.1
    num_target_tokens: int = 2

    # YOLO mask post-processing.
    mask_blur_kernel_size: int = 7
    mask_blur_sigma: float = 2.0

    def __post_init__(self):
        if self.mode not in {"adapter", "legacy_multiply"}:
            raise ValueError(f"Unknown MaskWeightConfig.mode={self.mode!r}.")
        if not 0.0 <= self.mask_dropout_p <= 1.0:
            raise ValueError("mask_dropout_p must be in [0, 1].")
        if self.adapter_hidden_dim <= 0:
            raise ValueError("adapter_hidden_dim must be positive.")
        if self.context_dilation < 0:
            raise ValueError("context_dilation must be non-negative.")
        if self.num_target_tokens < 0:
            raise ValueError("num_target_tokens must be non-negative.")
