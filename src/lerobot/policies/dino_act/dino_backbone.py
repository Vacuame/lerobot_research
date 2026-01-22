import torch
import torch.nn as nn
import torchvision.transforms as T
import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.customACT.configuration_customACT import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE, HIS_OBS_STATES
#新增：自己的import
from lerobot.policies.customACT.history_obs_state.modeling_history_obs import HistoryObsStateEmbedding
from lerobot.policies.dino_act.backbone_res import ResNet18Backbone, get_custom_backbone
class DinoV2Backbone(nn.Module):
    def __init__(self, model_size='small', freeze=True, target_dim=None):
        """
        参数:
            model_size: 'small', 'base', 'large', 'giant' (推荐 small 用于机器人)
            freeze: 是否冻结 DINOv2 的权重 (机器人数据少，建议冻结，只练下游)
            target_dim: 如果不为 None，会加一层 Linear 将特征投影到指定维度 (例如对齐 ResNet18 的 512)
        """
        super().__init__()
        
        # 1. 加载 Meta 官方 DINOv2 模型
        # 'dinov2_vits14' 对应 Small (推理速度最快，接近 ResNet50)
        # 'dinov2_vitb14' 对应 Base
        map_size = {
            'small': 'dinov2_vits14',
            'base':  'dinov2_vitb14',
            'large': 'dinov2_vitl14',
            'giant': 'dinov2_vitg14'
        }
        
        print(f"Loading DINOv2 ({model_size})...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', map_size[model_size])
        
        # 获取原始特征维度 (Small=384, Base=768)
        self.embed_dim = self.backbone.embed_dim
        
        # 2. 冻结权重 (建议)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()  # 设置为评估模式
            print("DINOv2 backbone is frozen.")
            
        # 3. 维度对齐 (可选)
        # 如果原本的策略网络期待 512 维输入 (ResNet18默认)，这里可以强行投影回去
        self.projection = None
        if target_dim and target_dim != self.embed_dim:
            self.projection = nn.Linear(self.embed_dim, target_dim)
            self.output_dim = target_dim
        else:
            self.output_dim = self.embed_dim

        # 4. 标准化 (DINOv2 需要 ImageNet 的均值方差)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
        self.downsample_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2d = nn.Conv2d(384, 512, kernel_size=1)#统一输出维度为512

    def forward(self, images):
    # 1. 归一化
        images = F.interpolate(images, size=(448, 588), mode='bilinear', align_corners=False)

        x = self.normalize(images)

        # 2. 提取所有特征 (包括 Patch Tokens)
        ret = self.backbone.get_intermediate_layers(x, n=1, reshape=True)
        # n=1 表示取最后一层, reshape=True 会自动把序列转回 [B, C, H, W]

        feature_map = ret[0] 
        # 对于 224x224 输入，DINOv2 (patch 14) 输出是 [B, 384, 16, 16]

        feature_map = self.downsample_pool(feature_map)
        

        feature_map = self.conv2d(feature_map)  # [B, 512, H, W]
        print("DINOv2 output shape after conv:", feature_map.shape)
        return {"feature_map": feature_map} # 现在输出是 [B, 512, 16, 16]，与 ResNet 结构一致了

# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟一个 Batch 的图片 (Batch=4, Channel=3, Height=224, Width=224)
    dummy_input = torch.randn(4, 3, 480, 640)
    print(f"Input1 shape: {dummy_input.shape}")


    # 实例化 (模仿 ResNet18 的 512 维输出)
    model1 = DinoV2Backbone(model_size='small', freeze=True, target_dim=512)
    output1 = model1(dummy_input)['feature_map']
    print(f"Output1 shape: {output1.shape}")




    config = ACTConfig()
    backbone_model = getattr(torchvision.models, config.vision_backbone)(
                    replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                    weights=config.pretrained_backbone_weights,
                    norm_layer=FrozenBatchNorm2d,
                )
    model2 = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
    output2 = model2(dummy_input)["feature_map"]
    print(f"Output2 shape: {output2.shape}") # 应该是 [4, 512, H', W']