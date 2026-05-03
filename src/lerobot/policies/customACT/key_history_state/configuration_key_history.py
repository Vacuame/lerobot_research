from dataclasses import dataclass, field


@dataclass
class KeyHistoryTokenConfig:
    """Config for event-guided key historical state tokens.

    This module only uses proprioceptive state history. It does not use
    action-state execution error, failure labels, or recovery scores.
    """

    enabled: bool = False
    history_len: int = 64
    num_segments: int = 4
    hidden_dim: int = 256
    conv_kernel_size: int = 3
    conv_dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dropout: float = 0.1
    prior_scale_init: float = 0.5
    selection_temperature: float = 1.0
    action_loss_weight: float = 0.05
    event_loss_weight: float = 0.01
    token_pos_embed: str = "learned"

