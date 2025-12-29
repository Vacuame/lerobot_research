
import numpy as np

def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return inter / union

class StableObjectManager:
    def __init__(
        self,
        iou_thresh=0.6,
        max_missing=5
    ):
        self.iou_thresh = iou_thresh
        self.max_missing = max_missing

        self.next_id = 0
        self.objects = {}  
        # stable_id -> {
        #   "mask": np.ndarray(bool),
        #   "missing": int
        # }

    def update(self, masks):
        """
        masks: List[np.ndarray(bool)]
        return: List[stable_id] (与 masks 顺序一致)
        """

        assigned = [-1] * len(masks)
        used_stable_ids = set()

        # 1. 尝试匹配已有对象（mask IoU）
        for i, mask in enumerate(masks):
            best_iou = 0
            best_sid = None

            for sid, obj in self.objects.items():
                if sid in used_stable_ids:
                    continue

                iou = mask_iou(mask, obj["mask"])
                if iou > best_iou:
                    best_iou = iou
                    best_sid = sid

            if best_iou > self.iou_thresh:
                assigned[i] = best_sid
                used_stable_ids.add(best_sid)
                self.objects[best_sid]["mask"] = mask
                self.objects[best_sid]["missing"] = 0

        # 2. 新对象 → 分配新 stable_id
        for i, sid in enumerate(assigned):
            if sid == -1:
                new_id = self.next_id
                self.next_id += 1

                self.objects[new_id] = {
                    "mask": masks[i],
                    "missing": 0
                }
                assigned[i] = new_id

        # 3. 未匹配对象 missing +1
        alive_ids = set(assigned)
        for sid in list(self.objects.keys()):
            if sid not in alive_ids:
                self.objects[sid]["missing"] += 1
                if self.objects[sid]["missing"] > self.max_missing:
                    del self.objects[sid]

        return assigned
