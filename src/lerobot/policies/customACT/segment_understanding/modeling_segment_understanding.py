# modeling.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.policies.customACT.segment_understanding.configuration_segment_understanding import SegmentUnderstandingConfig


# 主要embedding模块
class SegmentUnderstandingEmbedding(nn.Module):
    def __init__(self, config: SegmentUnderstandingConfig):
        super().__init__()
        self.r_encoder = REncoder(config)
        self.fk_encoder = FKEncoder(config)
        self.fusion = FusionMLP(config)

    def forward(self, R, R_mask, FK):
        e_R  = self.r_encoder(R, R_mask)   # [B, dim_model]
        e_FK = self.fk_encoder(FK)          # [B, dim_model]
        out  = self.fusion(e_R, e_FK)       # [B, dim_model]
        return out

# 对YOLO检测到的物体集合 R 进行编码
class REncoder(nn.Module):
    """
    输入:
        R: [B, N, 1 + r_numeric_dim]  # cls在第0维，后面是numeric
        R_mask: [B, N]                 # 物体存在 mask
    输出:
        e_R_global: [B, r_global_dim]
    """
    def __init__(self, config: SegmentUnderstandingConfig):
        super().__init__()
        self.config = config
        r_numeric_dim = 5 # 
        self.cls_embed = nn.Embedding(config.num_classes, config.cls_embed_dim)
        self.num_encoder = nn.Sequential(
            nn.Linear(r_numeric_dim, config.r_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.r_hidden_dim, config.r_hidden_dim),
            # 最后一层不加 ReLU，保持负值信息
        )
        self.token_proj = nn.Linear(
            config.cls_embed_dim + config.r_hidden_dim,
            config.r_token_dim
        )
        self.global_proj = nn.Linear(
            config.r_token_dim, 
            config.r_global_dim
            )

        self.pooling = config.pooling  # "mean" 或 "attention"
        if self.pooling == "attention":
            self.attn_query = nn.Parameter(torch.randn(config.r_token_dim))

    def forward(self, R, R_mask):
        cls_ids = R[..., 0].long()       # [B, N]
        r_num   = R[..., 1:]             # [B, N, r_numeric_dim]

        # 1. cls embedding
        e_cls = self.cls_embed(cls_ids)  # [B, N, cls_embed_dim]

        # 2. numeric features embedding
        e_num = self.num_encoder(r_num)  # [B, N, r_hidden_dim]

        # 3. 拼接与投影
        e_token = torch.cat([e_cls, e_num], dim=-1)  # [B, N, cls+hidden]
        e_token = self.token_proj(e_token)           # [B, N, r_token_dim]

        # 4. pooling
        if self.pooling == "mean":
            mask = R_mask.unsqueeze(-1).float()      # [B, N, 1]
            e_token = e_token * mask
            e_pooled = e_token.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, r_token_dim]
        elif self.pooling == "attention":
            scores = torch.einsum("bnd,d->bn", e_token, self.attn_query)   # [B, N]
            scores = scores.masked_fill(~R_mask, -1e9)
            attn = F.softmax(scores, dim=1)                                 # [B, N]
            e_pooled = torch.einsum("bn,bnd->bd", attn, e_token)           # [B, r_token_dim]
        else:
            raise ValueError("Unknown pooling type")

        # 5. 全局投影
        e_R_global = self.global_proj(e_pooled)  # [B, r_global_dim]

        return e_R_global


# FK 编码器，对FK进行简单处理即可（因为FK本身也是挺简单的数据）
# 将低维特征映射到嵌入空间的标准做法就是 2 层带 ReLU 的 MLP
class FKEncoder(nn.Module):
    def __init__(self, config: SegmentUnderstandingConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.fk_input_dim, config.fk_hidden_dim),
            nn.ReLU(),  # 激活函数用于引入非线性
            nn.Linear(config.fk_hidden_dim, config.fk_embed_dim),
        )

    def forward(self, fk):
        # fk: [B, fk_dim]
        return self.mlp(fk)  # [B, dim_model]


# R 和 FK 融合模块，输出最终 embedding（与图片特征拼合的那种） 
class FusionMLP(nn.Module):
    def __init__(self, config: SegmentUnderstandingConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                config.r_global_dim + config.fk_embed_dim,  # 将 R 和 FK 特征拼接
                config.fusion_hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                config.fusion_hidden_dim,
                config.output_dim  # 输出最终的 embedding
            ),
        )

    def forward(self, e_R, e_FK):
        x = torch.cat([e_R, e_FK], dim=-1)
        return self.mlp(x)  # [B, dim_model]
