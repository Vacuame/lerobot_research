from ultralytics import YOLO
import cv2
import torch
import torch.nn as nn


class yolo_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        yolo = YOLO("yolo11n-seg.pt")
        self.l1 = yolo.model.model[0]   
        self.l2 = yolo.model.model[1]
        self.l3 = yolo.model.model[2]
        self.l4 = yolo.model.model[3]
        self.l5 = yolo.model.model[4]
        self.l6 = yolo.model.model[5]
        self.l7 = yolo.model.model[6]
        self.l8 = yolo.model.model[7]
        self.l9 = yolo.model.model[8]
        self.l10 = yolo.model.model[9]
        self.l11 = yolo.model.model[10]
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        return x


# 输出模型结构
# model = YOLO("yolo11n-seg.pt")
# for i, m in enumerate(model.model.model):
#     print(i, m.__class__.__name__)



img = cv2.imread("custom/backbone/sheep.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. HWC -> CHW
img = img.transpose(2, 0, 1)

# 3. numpy -> torch
x = torch.from_numpy(img).float() / 255.0
x = x.unsqueeze(0).cuda()

# 4. 加载模型
yolo = YOLO("yolo11l-seg.pt")
yolo.model = yolo.model.cuda()
yolo.model.eval()

# 5. forward backbone
# with torch.no_grad():
#     feat = x
#     for i in range(11):  # 0~10
#         feat = yolo.model.model[i](feat)
model = yolo_backbone().cuda()
print(x.shape)
feat = model(x)

print("Backbone feature shape:", feat.shape)


