import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from lerobot.policies.customACT.segment_understanding.configuration_segment_understanding import (
    SegmentUnderstandingConfig,
)


class YoloDataProcessor:
    def __init__(self, config: SegmentUnderstandingConfig, device):
        self.config = config
        self.device = device
        self.max_objects = config.max_yolo_objects
        self.yolo = YOLO(config.yolo_path)
        self.ee_anchor = torch.tensor(config.ee_anchor, dtype=torch.float32, device=device)

    @torch.no_grad()
    def get_yolo_results(self, frames):
        # results = self.yolo.track(
        #     source=frames,
        #     persist=True,
        #     verbose=False,
        #     conf=0.25,
        #     iou=0.7,
        #     tracker=self.config.tracker_path,
        # )
        return self.yolo.predict(
            source=frames,
            verbose=False,
            conf=0.25,
            iou=0.7,
            device=self.device,
        )

    @torch.no_grad()
    def get_yolo_data(self, frames, mask_size: tuple[int, int] | None = None):
        results = self.get_yolo_results(frames)
        return self.get_yolo_data_from_results(results, mask_size=mask_size)

    @torch.no_grad()
    def get_yolo_data_from_results(self, results, mask_size: tuple[int, int] | None = None):
        object_features = []
        object_masks = []
        instance_masks = []

        for result in results:
            features_i, mask_i, instance_masks_i = self.process_single_result(result, mask_size=mask_size)
            object_features.append(features_i)
            object_masks.append(mask_i)
            if mask_size is not None:
                instance_masks.append(instance_masks_i)

        features = torch.stack(object_features, dim=0)
        masks = torch.stack(object_masks, dim=0)

        if mask_size is None:
            return features, masks

        return features, masks, torch.stack(instance_masks, dim=0)

    def process_single_result(self, result, mask_size: tuple[int, int] | None = None):
        device = self.device
        numeric_dim = self.config.object_numeric_dim
        feature_dim = 1 + numeric_dim

        features = torch.zeros(self.max_objects, feature_dim, dtype=torch.float32, device=device)
        valid_mask = torch.zeros(self.max_objects, dtype=torch.bool, device=device)
        feature_masks = None
        if mask_size is not None:
            feature_masks = torch.zeros(
                self.max_objects,
                1,
                mask_size[0],
                mask_size[1],
                dtype=torch.float32,
                device=device,
            )

        if result.boxes is None or len(result.boxes) == 0:
            return features, valid_mask, feature_masks

        boxes = result.boxes
        xywhn = boxes.xywhn.to(device=device, dtype=torch.float32)
        xyxyn = boxes.xyxyn.to(device=device, dtype=torch.float32)
        cls = boxes.cls.to(device=device, dtype=torch.float32)
        conf = boxes.conf.to(device=device, dtype=torch.float32)

        cx = xywhn[:, 0]
        cy = xywhn[:, 1]
        width = xywhn[:, 2].clamp(min=1e-6)
        height = xywhn[:, 3].clamp(min=1e-6)
        aspect = (width / height).clamp(max=10.0)

        obj_xy = torch.stack([cx, cy], dim=-1)
        delta = obj_xy - self.ee_anchor.to(device=device, dtype=torch.float32)
        dx = delta[:, 0]
        dy = delta[:, 1]
        dist = torch.norm(delta, dim=1)
        theta = torch.atan2(dy, dx)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        mask_area_norm = self._get_mask_area_norm(result, xywhn, device)

        # [cls, cx, cy, w, h, aspect, dx, dy, dist, sin, cos, conf, area]
        raw = torch.stack(
            [
                cls,
                cx,
                cy,
                width,
                height,
                aspect,
                dx,
                dy,
                dist,
                sin_theta,
                cos_theta,
                conf,
                mask_area_norm,
            ],
            dim=1,
        )

        relevance = (
            conf
            - float(self.config.anchor_distance_weight) * dist
            + float(self.config.anchor_area_weight) * mask_area_norm
        )
        order = torch.argsort(relevance, descending=True)
        raw = raw[order]
        xyxyn = xyxyn[order]

        n_objects = min(raw.shape[0], self.max_objects)
        features[:n_objects] = raw[:n_objects]
        valid_mask[:n_objects] = True

        if feature_masks is not None:
            masks = self._get_instance_masks(result, xyxyn, order, mask_size, device)
            feature_masks[:n_objects] = masks[:n_objects]

        return features, valid_mask, feature_masks

    def _get_mask_area_norm(self, result, xywhn: torch.Tensor, device: torch.device) -> torch.Tensor:
        if result.masks is None or len(result.masks.data) == 0:
            return xywhn[:, 2] * xywhn[:, 3]

        masks = result.masks.data.to(device=device, dtype=torch.float32)
        mask_area = masks.sum(dim=(1, 2))
        img_area = masks.shape[-1] * masks.shape[-2]
        return mask_area / max(float(img_area), 1.0)

    def _get_instance_masks(
        self,
        result,
        xyxyn: torch.Tensor,
        order: torch.Tensor,
        mask_size: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if result.masks is not None and len(result.masks.data) > 0:
            masks = result.masks.data.to(device=device, dtype=torch.float32)[order]
            return F.interpolate(
                masks.unsqueeze(1),
                size=mask_size,
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)

        height, width = mask_size
        masks = torch.zeros(xyxyn.shape[0], 1, height, width, dtype=torch.float32, device=device)
        scaled_boxes = xyxyn.clone()
        scaled_boxes[:, [0, 2]] *= width
        scaled_boxes[:, [1, 3]] *= height
        scaled_boxes = scaled_boxes.round().to(torch.int64)

        for i, (x1, y1, x2, y2) in enumerate(scaled_boxes.tolist()):
            x1 = max(0, min(width, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1))
            y2 = max(0, min(height, y2))
            if x2 > x1 and y2 > y1:
                masks[i, 0, y1:y2, x1:x2] = 1.0

        return masks

    @staticmethod
    def _tensor_to_uint8_hwc(frame: torch.Tensor) -> np.ndarray:
        """Convert CHW torch image in [0,1] to HWC uint8."""
        frame = frame.detach().to("cpu")
        if frame.ndim != 3:
            raise ValueError(f"Expected CHW image tensor, got shape={tuple(frame.shape)}")
        frame = frame.clamp(0, 1)
        frame = (frame * 255.0).to(torch.uint8)
        return frame.permute(1, 2, 0).numpy()

    def draw_results_on_frame(self, frame: np.ndarray, result) -> np.ndarray:
        """
        Draw masks/boxes on one RGB frame using one Ultralytics result.
        This mirrors custom/scripts/yolo/yolo_test.py::draw_results_on_frame.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return out

        boxes = result.boxes.xyxy.detach().to("cpu").numpy().astype(int)
        classes = result.boxes.cls.detach().to("cpu").numpy().astype(int)
        confs = result.boxes.conf.detach().to("cpu").numpy()
        track_ids = None if result.boxes.id is None else result.boxes.id.detach().to("cpu").numpy().astype(int)
        names = result.names if hasattr(result, "names") else {}

        if result.masks is not None:
            masks = result.masks.data.detach().to("cpu").numpy()
            for i, mask in enumerate(masks):
                if i >= len(boxes):
                    break
                mask = cv2.resize(mask, (w, h))
                mask = (mask > 0.5).astype(np.uint8)

                color = np.zeros_like(out)
                color[:, :, 1] = mask * 255
                out = cv2.addWeighted(out, 1.0, color, 0.4, 0)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cls_id = classes[i] if i < len(classes) else -1
            conf = confs[i] if i < len(confs) else 0.0
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            id_text = f"id:{track_ids[i]}, " if track_ids is not None and i < len(track_ids) else ""
            label = f"{id_text}{cls_name}, {conf:.2f}"

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return out

    def make_debug_overlay_chw(self, frame: torch.Tensor, result) -> np.ndarray:
        """Create CHW uint8 image for rerun logging."""
        frame_hwc = self._tensor_to_uint8_hwc(frame)
        overlay_hwc = self.draw_results_on_frame(frame_hwc, result)
        return np.transpose(overlay_hwc, (2, 0, 1))
