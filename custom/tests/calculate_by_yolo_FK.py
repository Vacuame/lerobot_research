
# 现在预设 末端在图片中是 0.5,1 (一半的宽度, 底部)

from ultralytics import YOLO
import cv2
import numpy as np
import torch


def get_seg_data(
    model_path,
    picture_path="custom/tests/yolo_test/test.jpg"
):
    # 加载模型
    model = YOLO(model_path)

    # 读取图片
    frame = cv2.imread(picture_path)
    if frame is None:
        print(f"Error: Could not read image from {picture_path}")
        return

    # YOLO 推理（直接喂 numpy）
    results = model(
            source=frame,
            verbose=False,  # 不打印日志
            conf=0.25,  # 检测置信度低于 0.25 的 bbox 会被丢弃
            iou=0.7,    # 重叠度大于 x 的 bbox 会被合并
            device=0,   # 使用 GPU 0
        )
    return results

def deal_with_data(results, ee_anchor):
    boxes = results.boxes
    xywhn = boxes.xywhn      # [N, 4]
    obj_xy = xywhn[:, :2]            # [N, 2]
    cls = boxes.cls          # [N]
    conf = boxes.conf        # [N]
    masks = results.masks    # [N, H, W]  (seg模型)

    delta = obj_xy - ee_anchor          # [N, 2]
    dist = torch.norm(delta, dim=1)  # [N]

    theta = torch.atan2(delta[:,1], delta[:,0])
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    if masks is not None:
        mask_area = masks.data.float().sum(dim=(1, 2))
        img_area = masks.data.shape[-1] * masks.data.shape[-2]
        mask_area_norm = mask_area / img_area
    else:
        mask_area_norm = xywhn[:, 2] * xywhn[:, 3]  # 没有的话就用norm方框代替

    # cls_emb = class_embedding(cls.long())  # [N, C] #以后再用吧

    tensors = [
        cls.unsqueeze(1),
        sin_theta.unsqueeze(1),
        cos_theta.unsqueeze(1),
        dist.unsqueeze(1),
        conf.unsqueeze(1),
        mask_area_norm.unsqueeze(1),
    ]
    R = torch.cat(tensors, dim=1)

    return R
    
if __name__ == "__main__":
    data = get_seg_data(model_path="yolo11l-seg.pt",picture_path="custom/scripts/yolo/image/sheep.jpg")
    for r in data:
        device = r.boxes.cls.device
        ee_anchor = torch.tensor([0.5, 1.0], device=device)
        R = deal_with_data(r,ee_anchor)
        torch.set_printoptions(sci_mode=False, precision=4, linewidth=120)
        print(R)

    
    