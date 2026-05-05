from dataclasses import dataclass


@dataclass
class AdaptiveActionChunkingConfig:
    """Inference-only adaptive action chunking settings.

    The policy still predicts the fixed ACT chunk. This controller only decides how many
    predicted actions to keep for execution, and how much old temporal predictions should
    influence the next action when temporal ensembling is enabled.
    """

    state_history_len: int = 64

    min_chunk_size: int = 8
    # 0 means "use the runtime upper bound", i.e. min(policy.chunk_size, policy.n_action_steps).
    max_chunk_size: int = 0
    stable_chunk_multiplier: float = 1.5
    unstable_chunk_multiplier: float = 0.5
    chunk_smoothing: float = 0.35
    max_chunk_delta: int = 16

    # These thresholds are measured on normalized proprioceptive state histories.
    volatility_low: float = 0.03
    volatility_high: float = 0.12
    acceleration_high: float = 0.10
    recovery_score_high: float = 0.65

    action_uncertainty_low: float = 0.03
    action_uncertainty_high: float = 0.15
    action_curvature_weight: float = 0.5

    stable_old_action_weight: float = 0.65
    nominal_old_action_weight: float = 0.35
    unstable_old_action_weight: float = 0.05
    min_old_action_weight: float = 0.0
    max_old_action_weight: float = 0.9
