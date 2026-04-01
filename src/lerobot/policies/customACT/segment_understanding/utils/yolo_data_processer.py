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
    def get_yolo_results(self,frames):
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
            device = self.device,
        )

    @torch.no_grad() # 使用YOLO时不计算梯度
    def get_yolo_data(self, frames):
        """
        frames: single image or list of images
        return:
            R:      [B, N_max, r_dim]
            R_mask: [B, N_max]
        """
       
        # results 是一个 list，长度 = batch_size
        results = self.get_yolo_results(frames)

        #DEBUG 打印结果
        # for r in results:
        #     if r.boxes is not None:
        #         for box in r.boxes:
        #             xyxy = box.xyxy[0].cpu().numpy()
        #             conf = box.conf.item()
        #             cls_id = int(box.cls.item())
        #             cls_name = r.names[cls_id]
        #             print(f"{cls_name}   at {xyxy}     conf={conf:.2f}")

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
        print("Make Debug Img")
        return np.transpose(overlay_hwc, (2, 0, 1))
