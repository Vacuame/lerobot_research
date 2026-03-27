import os
import torch
from PIL import Image

def save_img_list_compare(imgs, masks, save_dir):
    """
    极简版：保存图像和掩码
    imgs: [B, 3, H, W]
    masks: [B, 1, H, W]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 找到当前最大序号
    existing_files = [f for f in os.listdir(save_dir) if f.startswith("img_") and f.endswith(".png")]
    if existing_files:
        max_idx = max([int(f.split("_")[1].split(".")[0]) for f in existing_files])
    else:
        max_idx = 0
    
    B = imgs.shape[0]

    for i in range(B):
        idx = max_idx + i + 1
        
        # 图像转 PIL
        img_np = imgs[i].permute(1, 2, 0).cpu().numpy()
        img_pil = Image.fromarray((img_np * 255).astype('uint8'))
        
        # 掩码转 PIL（转 3 通道）
        mask_np = masks[i, 0].cpu().numpy()
        mask_pil = Image.fromarray((mask_np * 255).astype('uint8')).convert('RGB')
        
        # 保存原图
        img_pil.save(os.path.join(save_dir, f"img_{idx}.png"))
        
        # 保存掩码
        mask_pil.save(os.path.join(save_dir, f"mask_{idx}.png"))
        
        # 保存并排图（原图+掩码）
        combined = Image.new('RGB', (img_pil.width * 2, img_pil.height))
        combined.paste(img_pil, (0, 0))
        combined.paste(mask_pil, (img_pil.width, 0))
        combined.save(os.path.join(save_dir, f"combined_{idx}.png"))
        
        print(f"✅ 保存：img_{idx}.png, mask_{idx}.png, combined_{idx}.png")


import os
import torch
import numpy as np
from PIL import Image

def save_img_list(imgs, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    
    # 找到当前最大序号
    existing_files = [f for f in os.listdir(save_dir) if f.startswith("img_") and f.endswith(".png")]
    if existing_files:
        try:
            max_idx = max([int(f.split("_")[1].split(".")[0]) for f in existing_files])
        except:
            max_idx = 0
    else:
        max_idx = 0
    
    B = imgs.shape[0]

    for i in range(B):
        idx = max_idx + i + 1
        
        # 图像转 numpy [C, H, W] -> [H, W, C]
        img_np = imgs[i].permute(1, 2, 0).cpu().detach().numpy()
        
        # 数据处理：避免盲目 *255，根据数据范围自动判断
        # 如果已经是 uint8，直接使用；如果是 float 且范围在 0-1，则转 0-255
        if img_np.dtype == np.uint8:
            img_uint8 = img_np
            origin_file_format = 'uint8'
        else:
            # 浮点数情况
            if img_np.max() <= 1.0:
                # 假设是 [0, 1] 范围的浮点数
                origin_file_format = 'float_0_1'
                img_uint8 = (img_np * 255).astype('uint8')
            else:
                # 假设已经是 [0, 255] 范围的浮点数，直接转 uint8
                origin_file_format = 'float_0_255'
                img_uint8 = img_np.astype('uint8')
        
        # 处理通道数：PIL 对于单通道需要去掉最后一个维度 (H, W, 1) -> (H, W)
        if img_uint8.shape[-1] == 1:
            img_uint8 = img_uint8.squeeze(-1)
            mode = 'L'
        else:
            mode = 'RGB'
            
        img_pil = Image.fromarray(img_uint8, mode=mode)
        
        # 保存原图
        img_pil.save(os.path.join(save_dir, f"img_{idx}.png"))
        
        print(f"✅ 保存：img_{idx}.png, 原始格式: {origin_file_format}")