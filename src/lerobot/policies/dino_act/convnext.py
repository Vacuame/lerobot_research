import torchvision.models as models
import torch.nn as nn
import torch
class ConvNeXtBackbone(nn.Module):
    def __init__(self, target_dim=512):
        super().__init__()
        # 使用 convnext_tiny (性能优于 ResNet50，速度接近)
        # weights='DEFAULT' 会加载 ImageNet-1K 上的最新权重
        self.backbone = models.convnext_tiny(weights='DEFAULT')
        
        # 获取 ConvNeXt 的输出通道数 (Tiny版本通常是 768)
        self.in_features = self.backbone.classifier[2].in_features
        
        # 移除原本的分类头 (LayerNorm + Flatten + Linear)
        # ConvNeXt 的特征提取部分都在 .features 里
        self.backbone = self.backbone.features
        
        # 如果需要投影到 512 (适配 ACT)
        self.projection = None
        if target_dim and target_dim != self.in_features:
            # 使用 1x1 卷积来降维，保持特征图结构
            self.projection = nn.Conv2d(self.in_features, target_dim, kernel_size=1)
            # self.projection = CoordAwareProjection(self.in_features, target_dim) # 使用带坐标信息的投影

    def forward(self, x):
        # [B, 3, H, W] -> [B, 768, H/32, W/32]
        x = self.backbone(x)
        
        if self.projection:
            x = self.projection(x) # -> [B, 512, H/32, W/32]
        # print("convnext output shape:", x.shape)
        return {"feature_map": x}
    

class CoordAwareProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 输入通道 + 2 (一个是 x 坐标图，一个是 y 坐标图)
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
        batch, _, h, w = x.shape
        
        # 生成归一化的坐标网格 (-1 到 1)
        # y_grid: 上面是-1，下面是1
        # x_grid: 左边是-1，右边是1
        yy_channel = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(batch, 1, h, w)
        xx_channel = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(batch, 1, h, w)
        
        # 拼接到特征图上
        x_with_coords = torch.cat([x, xx_channel, yy_channel], dim=1)
        
        return self.conv(x_with_coords)

# 在主类中替换原本的 projection
# self.projection = CoordAwareProjection(768, 512)