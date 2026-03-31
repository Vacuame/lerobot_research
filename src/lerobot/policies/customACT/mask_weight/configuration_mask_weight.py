
from dataclasses import dataclass,field


@dataclass
class MaskWeightConfig:
    alpha: float = 1.0  # 融合权重