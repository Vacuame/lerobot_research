from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


def _to_numpy_hwc_image(image: np.ndarray | Tensor) -> np.ndarray:
    """Convert a single image to an HWC uint8 numpy array for detector inference."""
    if isinstance(image, Tensor):
        image = image.detach().cpu()
        if image.ndim != 3:
            raise ValueError(f"Expected a single image tensor with 3 dims, got shape {tuple(image.shape)}.")
        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.float().numpy()
    else:
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected a single image array with 3 dims, got shape {image.shape}.")

    if image.shape[-1] == 1:
        image = np.repeat(image, repeats=3, axis=-1)

    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0 + 1e-6 and image.min() >= -1e-6:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0)
    else:
        image = np.clip(image, 0, 255)

    return image.astype(np.uint8)


def _run_detector_single_image(detector: Any, image: np.ndarray) -> Any:
    if detector is None:
        return None

    if hasattr(detector, "predict"):
        try:
            return detector.predict(source=image, verbose=False)
        except TypeError:
            return detector.predict(image)

    if callable(detector):
        try:
            return detector(image, verbose=False)
        except TypeError:
            return detector(image)

    raise TypeError("detector must be None, callable, or expose a predict(...) method.")


def _normalize_names(names: Any) -> dict[int, str]:
    if names is None:
        return {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def _extract_candidate_detections(result: Any, detector: Any) -> list[dict[str, Any]]:
    if result is None:
        return []

    if isinstance(result, (list, tuple)):
        if not result:
            return []
        result = result[0]

    names = _normalize_names(getattr(result, "names", None))
    if not names:
        names = _normalize_names(getattr(detector, "names", None))

    if hasattr(result, "boxes") and result.boxes is not None:
        boxes = result.boxes
        if len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().tolist()
        conf = boxes.conf.detach().cpu().tolist() if getattr(boxes, "conf", None) is not None else None
        cls = boxes.cls.detach().cpu().tolist() if getattr(boxes, "cls", None) is not None else None

        candidates: list[dict[str, Any]] = []
        for idx, bbox in enumerate(xyxy):
            cls_id = int(cls[idx]) if cls is not None else None
            candidates.append(
                {
                    "bbox": [float(v) for v in bbox],
                    "conf": float(conf[idx]) if conf is not None else 1.0,
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id) if cls_id is not None else None),
                }
            )
        return candidates

    if isinstance(result, dict):
        boxes = result.get("boxes") or result.get("xyxy")
        scores = result.get("scores") or result.get("conf")
        labels = result.get("labels") or result.get("classes") or result.get("cls")
        names = _normalize_names(result.get("names")) or names

        if boxes is None:
            return []

        boxes = torch.as_tensor(boxes).detach().cpu().tolist()
        scores = torch.as_tensor(scores).detach().cpu().tolist() if scores is not None else [1.0] * len(boxes)
        labels = torch.as_tensor(labels).detach().cpu().tolist() if labels is not None else [None] * len(boxes)

        candidates = []
        for bbox, conf, cls_id in zip(boxes, scores, labels, strict=False):
            cls_id_int = int(cls_id) if cls_id is not None else None
            candidates.append(
                {
                    "bbox": [float(v) for v in bbox],
                    "conf": float(conf),
                    "class_id": cls_id_int,
                    "class_name": names.get(cls_id_int, str(cls_id_int) if cls_id_int is not None else None),
                }
            )
        return candidates

    return []


def detect_block_and_score(
    image: np.ndarray | Tensor,
    detector: Any,
    target_class_name: str = "block",
    edge_thresh: int = 10,
) -> dict[str, Any]:
    """
    Detect the highest-confidence target object in one view and convert it into a visibility score.
    """
    image_np = _to_numpy_hwc_image(image)
    height, width = image_np.shape[:2]
    image_area = max(height * width, 1)

    empty_result = {
        "detected": False,
        "conf": 0.0,
        "bbox": None,
        "area_ratio": 0.0,
        "border_score": 0.0,
        "score": 0.0,
    }

    if detector is None:
        return empty_result

    try:
        detector_output = _run_detector_single_image(detector, image_np)
        candidates = _extract_candidate_detections(detector_output, detector)
    except Exception:
        return empty_result

    target_class_name = target_class_name.lower()
    target_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("class_name", "")).lower() == target_class_name
    ]
    if not target_candidates:
        return empty_result

    best_candidate = max(target_candidates, key=lambda item: item["conf"])
    x1, y1, x2, y2 = best_candidate["bbox"]
    x1 = float(np.clip(x1, 0.0, width))
    y1 = float(np.clip(y1, 0.0, height))
    x2 = float(np.clip(x2, 0.0, width))
    y2 = float(np.clip(y2, 0.0, height))
    bbox = (x1, y1, x2, y2)

    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_ratio = float(bbox_area / image_area)
    border_score = float(
        x1 >= edge_thresh
        and y1 >= edge_thresh
        and (width - x2) >= edge_thresh
        and (height - y2) >= edge_thresh
    )
    conf = float(best_candidate["conf"])
    score = 0.6 * conf + 0.3 * area_ratio + 0.1 * border_score

    return {
        "detected": True,
        "conf": conf,
        "bbox": bbox,
        "area_ratio": area_ratio,
        "border_score": border_score,
        "score": float(score),
    }


def compute_view_weights(
    overall_result: dict[str, Any],
    robot1_result: dict[str, Any],
    eps: float = 1e-6,
) -> dict[str, float]:
    overall_score = float(overall_result.get("score", 0.0))
    robot1_score = float(robot1_result.get("score", 0.0))

    if overall_score <= 0.0 and robot1_score <= 0.0:
        return {
            "overall_weight": 0.5,
            "robot1_weight": 0.5,
        }

    total = overall_score + robot1_score + eps
    return {
        "overall_weight": float(overall_score / total),
        "robot1_weight": float(robot1_score / total),
    }


class AdaptiveFeatureFusion(nn.Module):
    def __init__(
        self,
        mode: str = "weighted_concat",
        concat_output_dim: int | None = None,
        input_feature_dim: int | None = None,
    ):
        super().__init__()
        if mode not in {"weighted_concat", "weighted_sum"}:
            raise ValueError(f"Unsupported fusion mode: {mode}")
        if mode == "weighted_concat" and concat_output_dim is not None and input_feature_dim is None:
            raise ValueError(
                "`input_feature_dim` must be provided when `concat_output_dim` is set for weighted_concat."
            )

        self.mode = mode
        self.concat_output_dim = concat_output_dim
        self.input_feature_dim = input_feature_dim
        self.concat_projection = (
            nn.Linear(input_feature_dim * 2, concat_output_dim)
            if self.mode == "weighted_concat" and concat_output_dim is not None
            else nn.Identity()
        )

    def _reshape_weight(self, weight: Tensor | float, feature: Tensor) -> Tensor:
        weight_tensor = torch.as_tensor(weight, device=feature.device, dtype=feature.dtype)
        if weight_tensor.ndim == 0:
            weight_tensor = weight_tensor.unsqueeze(0)
        while weight_tensor.ndim < feature.ndim:
            weight_tensor = weight_tensor.unsqueeze(-1)
        return weight_tensor

    def forward(
        self,
        overall_feature: Tensor,
        robot1_feature: Tensor,
        overall_weight: Tensor | float,
        robot1_weight: Tensor | float,
    ) -> Tensor:
        if overall_feature.shape[:-1] != robot1_feature.shape[:-1]:
            raise ValueError(
                "overall_feature and robot1_feature must match in every dim except the last one."
            )

        overall_weighted = overall_feature * self._reshape_weight(overall_weight, overall_feature)
        robot1_weighted = robot1_feature * self._reshape_weight(robot1_weight, robot1_feature)

        if self.mode == "weighted_sum":
            if overall_feature.shape != robot1_feature.shape:
                raise ValueError("weighted_sum requires overall_feature and robot1_feature to have identical shapes.")
            return overall_weighted + robot1_weighted

        fused_feature = torch.cat([overall_weighted, robot1_weighted], dim=-1)
        return self.concat_projection(fused_feature)


class VisibilityAwareMultiViewFusion(nn.Module):
    def __init__(
        self,
        detector: Any = None,
        target_class_name: str = "block",
        edge_thresh: int = 10,
        fusion_mode: str = "weighted_concat",
        concat_output_dim: int | None = None,
        input_feature_dim: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        object.__setattr__(self, "_detector", detector)
        self.target_class_name = target_class_name
        self.edge_thresh = edge_thresh
        self.eps = eps
        self.feature_fusion = AdaptiveFeatureFusion(
            mode=fusion_mode,
            concat_output_dim=concat_output_dim,
            input_feature_dim=input_feature_dim,
        )

    def set_detector(self, detector: Any) -> None:
        object.__setattr__(self, "_detector", detector)

    @property
    def detector(self) -> Any:
        return getattr(self, "_detector", None)

    def _split_batch_images(self, images: np.ndarray | Tensor | Sequence[Any]) -> tuple[list[Any], bool]:
        if isinstance(images, Tensor):
            if images.ndim == 3:
                return [images], True
            if images.ndim == 4:
                return list(images), False
        elif isinstance(images, np.ndarray):
            if images.ndim == 3:
                return [images], True
            if images.ndim == 4:
                return [images[idx] for idx in range(images.shape[0])], False
        elif isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
            return list(images), False

        raise ValueError("Images must be a single image, a batched tensor/array, or a sequence of images.")

    def _ensure_batched_feature(self, feature: Tensor, batch_size: int) -> tuple[Tensor, bool]:
        if feature.shape[0] == batch_size:
            return feature, False
        if batch_size == 1:
            return feature.unsqueeze(0), True
        raise ValueError(
            f"Feature batch dimension mismatch: expected {batch_size}, got first dim {feature.shape[0]}."
        )

    def forward(
        self,
        overall_image: np.ndarray | Tensor | Sequence[Any],
        robot1_image: np.ndarray | Tensor | Sequence[Any],
        overall_feature: Tensor,
        robot1_feature: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        overall_images, _ = self._split_batch_images(overall_image)
        robot1_images, _ = self._split_batch_images(robot1_image)
        if len(overall_images) != len(robot1_images):
            raise ValueError("overall_image and robot1_image batch sizes must match.")

        batch_size = len(overall_images)
        overall_feature, overall_unsqueezed = self._ensure_batched_feature(overall_feature, batch_size)
        robot1_feature, robot1_unsqueezed = self._ensure_batched_feature(robot1_feature, batch_size)
        if overall_unsqueezed != robot1_unsqueezed:
            raise ValueError("overall_feature and robot1_feature must agree on whether batch dim was implicit.")

        overall_results: list[dict[str, Any]] = []
        robot1_results: list[dict[str, Any]] = []
        overall_weights: list[float] = []
        robot1_weights: list[float] = []

        with torch.no_grad():
            for sample_idx in range(batch_size):
                overall_result = detect_block_and_score(
                    overall_images[sample_idx],
                    detector=self.detector,
                    target_class_name=self.target_class_name,
                    edge_thresh=self.edge_thresh,
                )
                robot1_result = detect_block_and_score(
                    robot1_images[sample_idx],
                    detector=self.detector,
                    target_class_name=self.target_class_name,
                    edge_thresh=self.edge_thresh,
                )
                weight_result = compute_view_weights(overall_result, robot1_result, eps=self.eps)

                overall_results.append(overall_result)
                robot1_results.append(robot1_result)
                overall_weights.append(weight_result["overall_weight"])
                robot1_weights.append(weight_result["robot1_weight"])

        overall_weight_tensor = torch.tensor(
            overall_weights, device=overall_feature.device, dtype=overall_feature.dtype
        )
        robot1_weight_tensor = torch.tensor(
            robot1_weights, device=robot1_feature.device, dtype=robot1_feature.dtype
        )
        fused_feature = self.feature_fusion(
            overall_feature=overall_feature,
            robot1_feature=robot1_feature,
            overall_weight=overall_weight_tensor,
            robot1_weight=robot1_weight_tensor,
        )

        if overall_unsqueezed:
            fused_feature = fused_feature.squeeze(0)

        debug_device = overall_feature.device
        debug_dtype = overall_feature.dtype
        debug_info = {
            "overall_score": torch.tensor(
                [result["score"] for result in overall_results], device=debug_device, dtype=debug_dtype
            ),
            "robot1_score": torch.tensor(
                [result["score"] for result in robot1_results], device=debug_device, dtype=debug_dtype
            ),
            "overall_weight": overall_weight_tensor,
            "robot1_weight": robot1_weight_tensor,
            "overall_detected": torch.tensor(
                [result["detected"] for result in overall_results], device=debug_device, dtype=torch.bool
            ),
            "robot1_detected": torch.tensor(
                [result["detected"] for result in robot1_results], device=debug_device, dtype=torch.bool
            ),
            "overall_bbox": [result["bbox"] for result in overall_results],
            "robot1_bbox": [result["bbox"] for result in robot1_results],
            "overall_result": overall_results,
            "robot1_result": robot1_results,
        }

        return fused_feature, debug_info
