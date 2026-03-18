# resnet18_custom.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Callable, Optional, List, Dict

# -------------------------
# BasicBlock（ResNet-18 / ResNet-34 使用）
# -------------------------
class BasicBlock(nn.Module):
    expansion: int = 1  # BasicBlock 不扩展通道

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        # 第一个 3x3 卷积（可能下采样）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个 3x3 卷积（stride=1）
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 下采样（shortcut）：当维度或尺寸变化时使用 1x1 conv
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# -------------------------
# ResNet 主体
# -------------------------
class ResNet(nn.Module):
    def __init__(self, block: Type[BasicBlock], layers: List[int], num_classes: int = 1000, zero_init_residual: bool = False):
        """
        block: BasicBlock
        layers: 每个 layer (stage) 的 block 数量，例如 [2,2,2,2] 对应 ResNet-18
        num_classes: 最后分类的类别数（如果作为 backbone 想去掉 fc，可忽略）
        """
        super().__init__()
        self.in_channels = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # -> 64 x 112 x 112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # -> 64 x 56 x 56

        # 4 个 stage（Layer1..Layer4）
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)   # -> 64 x 56 x 56
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # -> 128 x 28 x 28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # -> 256 x 14 x 14
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # -> 512 x 7 x 7

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 参数初始化（参考 torchvision 实现）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 可选：把最后一个 BN 的权重初始化为 0，使残差块初始为 0（improves training）
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[BasicBlock], out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """
        构建每个 stage（layer），blocks 是该 stage 的 block 数量。
        第一个 block 可能 stride=2 做下采样，且需要设置 downsample（1x1 conv）
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # 使用 1x1 conv 改变通道数并下采样
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 第一个 block（可能会下采样）
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels * block.expansion
        # 其余 block（stride=1）
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        return_features=False: 标准前向，返回 logits（分类）
        return_features=True: 返回一个 dict，包含多个中间特征（C2, C3, C4, C5）和 pooled vector
          - C2: layer1 输出，shape [B, 64, 56, 56]
          - C3: layer2 输出,  [B,128,28,28]
          - C4: layer3 输出,  [B,256,14,14]
          - C5: layer4 输出,  [B,512,7,7]
          - pooled: GAP -> [B,512]
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # -> C2 input

        # Stages
        c2 = self.layer1(x)  # -> [B,64,56,56]
        c3 = self.layer2(c2) # -> [B,128,28,28]
        c4 = self.layer3(c3) # -> [B,256,14,14]
        c5 = self.layer4(c4) # -> [B,512,7,7]

        if return_features:
            pooled = torch.flatten(self.avgpool(c5), 1)  # -> [B,512]
            return {"C2": c2, "C3": c3, "C4": c4, "C5": c5, "pooled": pooled}

        # 分类 head
        out = self.avgpool(c5)
        out = torch.flatten(out, 1)
        logits = self.fc(out)
        return logits

# -------------------------
# 工厂函数：resnet18
# -------------------------
def resnet18(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> ResNet:
    """
    返回一个 ResNet-18 实例（若需要预训练权重，请自行加载 torchvision 的权重）
    layers = [2,2,2,2] 对应 ResNet-18
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
    if pretrained:
        # 注意：本函数不在内部加载权重；如果需要可用 torchvision.models.resnet18(pretrained=True)
        raise NotImplementedError("预训练权重需通过 torchvision 加载或手动提供。")
    return model

# -------------------------
# 简易示例：打印结构与特征尺寸
# -------------------------
if __name__ == "__main__":
    # 创建模型（分类）
    model = resnet18(num_classes=1000)
    print("模型参数量（约）: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    # 随机输入（批大小 2，RGB 224x224）
    x = torch.randn(2, 3, 224, 224)
    # 标准前向（分类）
    logits = model(x)
    print("logits shape:", logits.shape)  # -> [2,1000]

    # 作为 backbone，获取多层特征
    feats = model(x, return_features=True)
    for k, v in feats.items():
        print(f"{k} shape: {v.shape}")
    # C2: [2,64,56,56]; C3: [2,128,28,28]; C4: [2,256,14,14]; C5: [2,512,7,7]; pooled: [2,512]
