import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class mymodule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3,3,kernel_size=7,stride=2,padding=3)
        self.layer2 = nn.BatchNorm2d(3)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x




# 1. 读取图片
img = Image.open("pic.jpg").convert("RGB")

# 2. 转成 Tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
img_tensor = transform(img).unsqueeze(0)   # shape: [1, 3, 224, 224]

print("img_tensor.shape:"+str(img_tensor.shape))

# 3. 定义一个卷积层（示例）
# 输出通道=1, 输入通道=3, 卷积核=3×3
model = mymodule()

# 4. 进行卷积运算
with torch.no_grad():
    feature = model(img_tensor)

print("feature.shape:"+str(feature.shape))

# 5. 取出第一个通道
feature_img = feature.squeeze().numpy()

feature_img = feature_img.transpose(1, 2, 0)

# 6. 归一化到 0-1，便于显示
feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())

# 7. 显示卷积后的图像
plt.imshow(feature_img)
plt.title("Convolution Result")
plt.axis("off")
plt.show()

# 8. 保存
