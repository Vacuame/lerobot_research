import torch

def pad_list_left_to_length(target_tensor: torch.Tensor, target_length: int) -> torch.Tensor:

    current_length = target_tensor.size(0)
    if current_length == target_length:
        return target_tensor
    elif current_length < target_length:
        # 不足：用第一个元素在前面重复填充
        num_pad = target_length - current_length
        first = target_tensor[0:1]  # 保持维度：(1, D)
        padding = first.repeat(num_pad, 1)  # (num_pad, D)
        padded = torch.cat([padding, target_tensor], dim=0)  # (target_length, D)
        return padded
    else:
        return target_tensor[-target_length:]