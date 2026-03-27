import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from typing import List, Union

def yolo_result_to_soft_mask(
    results: Union[List, object], 
    kernel_size: int = 7, 
    sigma: float = 2.0
):
    # ================= 1. 统一转为列表格式 =================
    if not isinstance(results, list):
        results = [results]
    
    B = len(results)
    
    # ================= 2. 从第一个 result 获取图像尺寸 =================
    # YOLO result 对象包含 orig_shape 属性 (H, W)
    H, W = results[0].orig_shape
    
    # ================= 3. 批量生成掩码 =================
    soft_mask_list = []
    
    for result in results:
        # --- 自动检测设备 ---
        if result.masks is not None:
            device = result.masks.data.device
        elif result.boxes is not None:
            device = result.boxes.conf.device
        else:
            device = torch.device('cpu')
        
        # --- 初始化硬掩码 ---
        hard_mask = torch.zeros((1, H, W), dtype=torch.float32, device=device)
        
        # --- 实例分割掩码优先 ---
        if result.masks is not None:
            masks = result.masks.data          # [N, h, w]
            confs = result.boxes.conf          # [N]
            
            for i in range(len(masks)):
                # 插值到原始图像尺寸
                mask_i = F.interpolate(
                    masks[i].unsqueeze(0).unsqueeze(0).float(),  # [1, 1, h, w]
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)          # [H, W]
                
                conf_i = confs[i].item()
                weighted_mask = mask_i * conf_i
                hard_mask[0] = torch.maximum(hard_mask[0], weighted_mask)
        
        # --- 或使用边界框 ---
        elif result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                conf = boxes.conf[i].item()
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                conf_tensor = torch.tensor(conf, device=device)
                hard_mask[0, y1:y2, x1:x2] = torch.maximum(
                    hard_mask[0, y1:y2, x1:x2],
                    conf_tensor
                )
        
        # --- 高斯模糊平滑 ---
        if hard_mask.max() > 0:
            blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            soft_mask = blur(hard_mask.unsqueeze(0))  # [1, 1, H, W]
            soft_mask = soft_mask.squeeze(0)          # [1, H, W]
            soft_mask = torch.clamp(soft_mask, 0, 1)
        else:
            soft_mask = hard_mask
        
        soft_mask_list.append(soft_mask)
    
    # ================= 4. 堆叠成 Batch =================
    soft_masks = torch.stack(soft_mask_list, dim=0)  # [B, 1, H, W]
    
    return soft_masks