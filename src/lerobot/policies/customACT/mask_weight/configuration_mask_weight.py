
from dataclasses import dataclass,field


@dataclass
class MaskWeightConfig:
    alpha: float = 1.0  # 前景增强
    beta: float  = 0.5 # 背景抑制