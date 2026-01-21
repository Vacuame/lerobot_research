from ultralytics import YOLO
import cv2
import numpy as np
import torch
from lerobot.policies.customACT.segment_understanding.configuration_segment_understanding import SegmentUnderstandingConfig

class YoloDataProcessor:
    def __init__(self, config: SegmentUnderstandingConfig, device):
        self.config = config
        self.device = device
        self.max_objects = config.max_yolo_objects
        self.yolo = YOLO(config.yolo_path)
        self.ee_anchor = torch.tensor([0.5, 1.0],device=device)  # 末端锚点位置

    @torch.no_grad() # 使用YOLO时不计算梯度
    def get_yolo_data(self, frames):
        """
        frames: single image or list of images
        return:
            R:      [B, N_max, r_dim]
            R_mask: [B, N_max]
        """

        # 反归一化到 [0, 1] 范围，因为act的图片经过了mean-std归一化，不符合YOLO需求
        frames = self.denormalize_with_imagenet_stats(frames)

        # results = self.yolo.track(
        #     source=frames,
        #     persist=True,
        #     verbose=False,
        #     conf=0.25,
        #     iou=0.7,
        #     tracker=self.config.tracker_path,
        # )

        results = self.yolo.predict(
            source=frames,
            verbose=False,
            conf=0.25,
            iou=0.7,
            device = self.device,
        )

        # results 是一个 list，长度 = batch_size
        R_list = []
        mask_list = []

        for res in results:
            R_i, mask_i = self.process_single_result(res)
            R_list.append(R_i)
            mask_list.append(mask_i)

        R = torch.stack(R_list, dim=0)        # [B, N_max, r_dim]
        R_mask = torch.stack(mask_list, dim=0)  # [B, N_max]
        
        return R, R_mask

    def process_single_result(self, result):
        device = self.device
        ee_anchor = self.ee_anchor.to(device)

        if result.boxes is None or len(result.boxes) == 0:
            R = torch.zeros(self.max_objects, 6, device=device)
            R_mask = torch.zeros(self.max_objects, dtype=torch.bool, device=device)
            return R, R_mask

        boxes = result.boxes
        xywhn = boxes.xywhn            # [N, 4]
        obj_xy = xywhn[:, :2]          # [N, 2]
        cls = boxes.cls                # [N]
        conf = boxes.conf              # [N]
        masks = result.masks           # [N, H, W] or None

        delta = obj_xy - ee_anchor     # [N, 2]
        dist = torch.norm(delta, dim=1)

        theta = torch.atan2(delta[:, 1], delta[:, 0])
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        if masks is not None:
            mask_area = masks.data.float().sum(dim=(1, 2))
            img_area = masks.data.shape[-1] * masks.data.shape[-2]
            mask_area_norm = mask_area / img_area
        else:
            mask_area_norm = xywhn[:, 2] * xywhn[:, 3]

        # [N, 6]
        R_raw = torch.stack(
            [
                cls,
                sin_theta,
                cos_theta,
                dist,
                conf,
                mask_area_norm,
            ],
            dim=1,
        )

        N = min(R_raw.shape[0], self.max_objects)

        R = torch.zeros(self.max_objects, 6, device=device)
        R_mask = torch.zeros(self.max_objects, dtype=torch.bool, device=device)

        R[:N] = R_raw[:N]
        R_mask[:N] = True

        return R.to(device), R_mask.to(device)

    def denormalize_with_imagenet_stats(self, normalized_img):
        mean = torch.tensor([0.485, 0.456, 0.406], device=normalized_img.device).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=normalized_img.device).view(-1, 1, 1)
        img = normalized_img * std + mean
        return torch.clamp(img, 0.0, 1.0)