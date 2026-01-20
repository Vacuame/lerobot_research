from ultralytics import YOLO
import cv2
import numpy as np
import torch
from lerobot.policies.customACT.segment_understanding.configuration_segment_understanding import SegmentUnderstandingConfig

class YoloDataProcessor:
    def __init__(self, config: SegmentUnderstandingConfig):
        self.config = config
        self.yolo = YOLO(config.yolo_path)
        self.ee_anchor = torch.tensor([0.5, 1.0])  # 末端锚点位置
    
    def get_data_from_yolo(self, frame):
        results = self.yolo.track(
            source=frame,
            persist=True,   # 跨帧保留 tracker，不会每帧都重置
            verbose=False,  # 不打印日志
            conf=0.25,  # 检测置信度低于 0.25 的 bbox 会被丢弃
            iou=0.7,    # 重叠度大于 x 的 bbox 会被合并
            tracker=self.config.tracker_path  # 指定用 BoT-SORT
        )
        return self.data_process(results, self.ee_anchor)


    def data_process(self, results, ee_anchor):
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

        # cls_emb = class_embedding(cls.long())  # [N, C] # 以后在模型里用吧

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