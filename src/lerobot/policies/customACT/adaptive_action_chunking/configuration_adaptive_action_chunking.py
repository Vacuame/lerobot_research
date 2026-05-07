from dataclasses import dataclass


@dataclass
class AdaptiveActionChunkingConfig:
    """推理阶段使用的历史感知自适应动作块配置。

    ACT 模型本身仍然预测固定长度的动作块。本配置只控制推理阶段保留多少步动作
    用于执行，以及在 temporal ensembling 时旧预测应该占多大权重。
    """

    # 用于计算历史状态波动的在线状态窗口长度。这里保存最近多少帧 observation.state。
    state_history_len: int = 64

    # 自适应动作块的最小执行长度。即使判断为不稳定，也不会低于这个长度。
    min_chunk_size: int = 8
    # 自适应动作块的最大执行长度。0 表示使用运行时上限：min(policy.chunk_size, policy.n_action_steps)。
    max_chunk_size: int = 0
    # 稳定阶段的动作块放大倍率。状态稳定时 K_final = K_aac * stable_chunk_multiplier。
    stable_chunk_multiplier: float = 1.5
    # 不稳定阶段的动作块缩短倍率。状态波动、动作不确定或 recovery_score 高时使用。
    unstable_chunk_multiplier: float = 0.5
    # 相邻两次自适应 chunk 长度的指数平滑系数。越大越快跟随新决策，越小越平滑。
    chunk_smoothing: float = 0.35
    # 单次更新允许 chunk 长度变化的最大步数，防止 K_final 在相邻推理间跳变过大。
    max_chunk_delta: int = 16

    # 状态低波动阈值。低于该值时，且其它指标也稳定，倾向于放大动作块。
    volatility_low: float = 0.03
    # 状态高波动阈值。高于该值时，认为环境或机器人状态变化较大，倾向于缩短动作块。
    volatility_high: float = 0.12
    # 状态加速度高阈值。历史状态变化突然加快时，认为可能发生接触、卡顿或阶段切换。
    acceleration_high: float = 0.10
    # recovery_score 高阈值。高于该值时，认为存在恢复/异常风险，倾向于缩短动作块。
    recovery_score_high: float = 0.65

    # 动作块低不确定性阈值。ACT 预测动作变化很平滑时，倾向于允许更长执行块。
    action_uncertainty_low: float = 0.03
    # 动作块高不确定性阈值。预测动作变化或曲率较大时，倾向于缩短执行块。
    action_uncertainty_high: float = 0.15
    # 动作曲率在不确定性计算中的权重。越大越重视动作序列二阶变化。
    action_curvature_weight: float = 0.5

    # 稳定阶段 temporal ensembling 中旧预测动作的权重。越大越相信历史预测。
    stable_old_action_weight: float = 0.65
    # 普通阶段 temporal ensembling 中旧预测动作的权重。
    nominal_old_action_weight: float = 0.35
    # 不稳定阶段 temporal ensembling 中旧预测动作的权重。越小越依赖当前新预测。
    unstable_old_action_weight: float = 0.05
    # 旧预测权重的下限，用于限制动态权重不会低于该值。
    min_old_action_weight: float = 0.0
    # 旧预测权重的上限，用于限制动态权重不会高于该值。
    max_old_action_weight: float = 0.9

    # 是否实时打印 AAC 调试信息，包括预测 chunk 长度、实际执行 chunk 长度和关键指标。
    debug_print_chunks: bool = True
    # 调试信息打印频率。1 表示每次新预测都打印，5 表示每 5 次预测打印一次。
    debug_print_every: int = 1
    # 调试时预览多少个动作步。只影响终端输出，不影响控制逻辑。
    debug_print_num_actions: int = 3
    # 调试时每个动作最多预览多少个维度。动作维度很多时用于避免刷屏。
    debug_print_action_dims: int = 6
