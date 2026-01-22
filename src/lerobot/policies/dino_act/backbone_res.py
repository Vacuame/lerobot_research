import torch
import torch.nn as nn
from torchvision.ops import FrozenBatchNorm2d  # 导入 Torchvision 自带的 FrozenBN

# -----------------------------------------------------------------------
# 1. 基础组件：3x3 和 1x1 卷积定义
# -----------------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# -----------------------------------------------------------------------
# 2. 基础模块：BasicBlock (ResNet18/34 使用这个，而不是 Bottleneck)
# -----------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1  # BasicBlock 的输出通道数等于输入通道数

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # ResNet BasicBlock 不支持 groups != 1 或 base_width != 64
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        
        # 第一层卷积
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二层卷积
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果输入输出维度不一致（比如stride=2或者通道数变了），需要通过 downsample 调整 identity
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# -----------------------------------------------------------------------
# 3. 主模型：ResNet18 Backbone (展开版)
# -----------------------------------------------------------------------
class ResNet18Backbone(nn.Module):
    def __init__(self, replace_stride_with_dilation=[False, False, False], norm_layer=FrozenBatchNorm2d):
        super(ResNet18Backbone, self).__init__()
        
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        
        # --- 初始层 ---
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- ResNet Layers ---
        self.layer1 = self._make_layer(BasicBlock, 64, blocks=2, stride=1)
        
        # Layer 2
        stride_layer2 = 2 if not replace_stride_with_dilation[0] else 1
        self.layer2 = self._make_layer(BasicBlock, 128, blocks=2, stride=stride_layer2,
                                       dilate=replace_stride_with_dilation[0])

        # Layer 3
        stride_layer3 = 2 if not replace_stride_with_dilation[1] else 1
        self.layer3 = self._make_layer(BasicBlock, 256, blocks=2, stride=stride_layer3,
                                       dilate=replace_stride_with_dilation[1])

        # Layer 4 (根据 config 决定是否使用空洞卷积)
        stride_layer4 = 2 if not replace_stride_with_dilation[2] else 1
        self.layer4 = self._make_layer(BasicBlock, 512, blocks=2, stride=stride_layer4,
                                       dilate=replace_stride_with_dilation[2])

        # 【重要改动】这里去掉了 self.avgpool 和 self.fc
        # 因为 ACT 策略只需要特征图，不需要分类层
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, FrozenBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # _make_layer 函数保持不变，这里省略不写，请保留之前的代码
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 
                            groups=1, base_width=64, dilation=previous_dilation, 
                            norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=1,
                                base_width=64, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 1. 前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 2. 【关键修复】不要做 avgpool/flatten，直接返回字典
        # 此时 x 的维度是 (Batch, 512, H', W')
        return {"feature_map": x}

# -----------------------------------------------------------------------
# 4. 实例化与权重加载 (模拟你的 getattr 行为)
# -----------------------------------------------------------------------
def get_custom_backbone(config):
    # 解析你的参数
    replace_strides = [False, False, config.replace_final_stride_with_dilation]
    
    # 初始化我们自定义的类
    model = ResNet18Backbone(
        replace_stride_with_dilation=replace_strides,
        norm_layer=FrozenBatchNorm2d
    )




    
    # 加载预训练权重 (如果需要)
    if config.pretrained_backbone_weights:
        print("Loading pretrained weights...")
        # 这里使用 torchvision 的标准权重加载逻辑
        from torchvision.models import ResNet18_Weights
        
        # 获取标准 ResNet18 权重
        state_dict = ResNet18_Weights.DEFAULT.get_state_dict(progress=True)
        
        # 因为我们的类结构和官方完全一致，可以直接加载 state_dict
        # strict=False 是为了防止 FrozenBN 和普通 BN 在某些 buffer 上的微小差异（虽然通常兼容）
        model.load_state_dict(state_dict, strict=False)
        
    return model